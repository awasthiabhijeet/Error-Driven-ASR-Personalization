import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import os
import argparse
import random
from math import floor,ceil
from functools import partial
from shutil import copyfile

from model import ErrorClassifierPhoneBiLSTM_V2
from data import error_classifier_collate_fn, geneate_error_data_from_hypotheses_file
from metrics import xent_loss, error_classifier_errors, get_precision_recall_f1
from helpers import print_dict, warmup_decay_policy, save_model

def eval(eval_data_loader,model,args,device):
  with torch.no_grad():
    total_errors = 0
    total_valid = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for data in eval_data_loader:
      data = [item.to(device) for item in data]
      phonemes, error_positions, padding_positions, sequence_lengths = data
      model.eval()
      logits = model(phonemes,sequence_lengths)
      dev_loss = xent_loss(logits, error_positions,padding_positions)
      predictions = torch.argmax(logits,-1)
      errors, true_positives, false_positives, false_negatives = error_classifier_errors(predictions,
                                                                   error_positions, padding_positions)
      num_valid_positions = torch.sum(padding_positions)  
      total_errors += errors
      total_valid += num_valid_positions

      total_true_positives += true_positives
      total_false_negatives += false_negatives
      total_false_positives += false_positives

    error_rate = (total_errors/total_valid) * 100
    precision,recall,f1 = get_precision_recall_f1(total_true_positives, total_false_positives, total_false_negatives)
  return dev_loss.item(), error_rate, precision, recall, f1

def train(
        train_data_loader,
        dev_data_loader,
        model,
        optimizer,
        args=None,
        fn_lr_policy=None,
        device=torch.device('cpu'),
        other_inputs=None):
  steps = 0
  print('pre evaluation...')
  dev_loss, dev_error_rate, precision, recall, f1 = eval(dev_data_loader,model,args,device)
  best_f1 = f1
  print('pre evaluation f1: {:.2f}'.format(f1))
  for epoch in range(1,args.num_epochs+1):
    for data in train_data_loader:
      steps +=1
      data = [item.to(device) for item in data] 
      phonemes, error_positions, padding_positions, sequence_lengths = data

      if fn_lr_policy is not None:
        adjusted_lr = fn_lr_policy(steps)
        for param_group in optimizer.param_groups:
          param_group['lr'] = adjusted_lr
      
      optimizer.zero_grad()
      model.train()
      logits = model(phonemes,sequence_lengths)
      predictions = torch.argmax(logits,-1)
      train_errors, true_positives, false_positives, false_negatives = error_classifier_errors(predictions,
                                                                         error_positions, padding_positions)
      precision, recall, f1 = get_precision_recall_f1(true_positives, false_positives, false_negatives)
      num_valid_positions = torch.sum(padding_positions)
      train_error_rate = (train_errors/num_valid_positions)*100
      train_loss = xent_loss(logits, error_positions,padding_positions)
      train_loss.backward()
      optimizer.step()

      if steps % args.train_frequency == 0:
        print('\t epoch: {} steps: {} train_precision: {:.2f} train_recall: {:.2f} train_f1: {:.2f} train_loss: {:.2f}\n'.format(epoch,steps,precision,recall,f1,train_loss))
        other_inputs["summary_writer"].add_scalar('Loss/train', train_loss.item(), steps)
        other_inputs["summary_writer"].add_scalar('Error/train', train_error_rate, steps)
        other_inputs["summary_writer"].add_scalar('precision/train', precision, steps)
        other_inputs["summary_writer"].add_scalar('recall/train', recall, steps)
        other_inputs["summary_writer"].add_scalar('f1/train', f1, steps)
      
    dev_loss, dev_error_rate, precision, recall, f1 = eval(dev_data_loader,model,args,device)
    other_inputs["summary_writer"].add_scalar('Loss/dev', dev_loss, steps)
    other_inputs["summary_writer"].add_scalar('Error/dev', dev_error_rate, steps)
    other_inputs["summary_writer"].add_scalar('precision/dev', precision , steps)
    other_inputs["summary_writer"].add_scalar('recall/dev', recall , steps)
    other_inputs["summary_writer"].add_scalar('f1/dev', f1 , steps)
    save_model(model,optimizer,epoch,output_dir=args.output_dir)

    if best_f1 is None or f1 > best_f1:
      best_f1 = f1
      if recall != 100.00: 
        # recall=100% imples model is predicting everything as error
        # and thus model is not not well trained yet.
        save_model(model, optimizer, epoch, output_dir=args.best_dir, save_optimizer=False)

    print('epoch: {} steps: {} best_f1: {:.2f} dev_precision: {:.2f} dev_recall: {:.2f} dev_f1: {:.2f} dev_loss: {:.2f}\n'.format(epoch,steps,
                                                              best_f1,precision, recall, f1, dev_loss))

  if not os.path.isdir(args.best_dir):
    # if no best ckpt is saved, copy the most recent ckpt
    print('no best ckpt was found, so copying ckpt from recent to best dir')
    class_name = model.__class__.__name__
    file_name = "{0}.pt".format(class_name)
    os.makedirs(args.best_dir)
    copyfile(os.path.join(args.output_dir,file_name),os.path.join(args.best_dir,file_name))

def parse_args():
    parser = argparse.ArgumentParser(description='phonebilstm')
    parser.add_argument("--batch_size", default=1, type=int, help='data batch size')
    parser.add_argument("--num_epochs", default=200, type=int, help='number of training epochs. if number of steps if specified will overwrite this')
    parser.add_argument("--train_freq", dest="train_frequency", default=20, type=int, help='number of iterations until printing training statistics on the past iteration')
    parser.add_argument("--lr", default=3e-4, type=float, help='learning rate')
    parser.add_argument("--weight_decay", default=1e-3, type=float, help='weight decay rate')
    parser.add_argument("--hypotheses_path", type=str, required=True, help='path to output file containing references and hypothesis generated by ASR model')
    parser.add_argument("--lr_decay", type=str, default='none', choices=['warmup','decay','none'], help='learning rate decay strategy')
    parser.add_argument("--output_dir", type=str, required=True, help='saves results in this directory')
    parser.add_argument("--best_dir", type=str, required=True, help='saves the best ckpt in this directory')
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--train_portion",default=0.65,type=float,help='portion of data used for training error model, rest is used as dev')
    parser.add_argument("--pretrained_ckpt", default=None, type=str, help='path to pretrained ckpt')
    parser.add_argument("--input_size",default=64,type=int,help='size of phone embeddings')
    parser.add_argument("--hidden_size",default=64,type=int,help='size of hidden cells of lstm')
    parser.add_argument("--num_layers",default=4,type=int,help='number of lstm layers')
    args=parser.parse_args()
    return args

def main(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  hypotheses_path = args.hypotheses_path

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  print('loading data....')
  data = geneate_error_data_from_hypotheses_file(hypotheses_path)
  random.shuffle(data)
  
  train_data = data[0:floor(args.train_portion*len(data))]
  dev_data = data[len(train_data):]

  print('train_size: {} dev_size: {}'.format(len(train_data),len(dev_data)))

  print('creating data loaders...')

  train_data_loader = torch.utils.data.DataLoader(train_data, 
                                                  batch_size=args.batch_size, 
                                                  shuffle=True, 
                                                  collate_fn=error_classifier_collate_fn, 
                                                  drop_last=False)

  dev_data_loader = torch.utils.data.DataLoader(dev_data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                collate_fn=error_classifier_collate_fn,
                                                drop_last=False)

  print('creating model...')
  phone_bilstm = ErrorClassifierPhoneBiLSTM_V2(input_size=args.input_size,hidden_size=args.hidden_size,num_layers=args.num_layers)
  if args.pretrained_ckpt:
    # optionally initialize an with an error model trained on errors of native speech
    # expected this to provide a warm start
    # however, did not observe any significant gains or losses from this step.
    print('loading pretrained cpkt from : {}'.format(args.pretrained_ckpt))
    checkpoint = torch.load(args.pretrained_ckpt, map_location="cpu")
    # del checkpoint['state_dict']['fc_layer_2.weight'] 
    # del checkpoint['state_dict']['fc_layer_2.bias']
    phone_bilstm.load_state_dict(checkpoint['state_dict'],strict=True)
    
  phone_bilstm.to(device)
  optimizer = optim.Adam(phone_bilstm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

  steps_per_epoch = ceil(len(train_data)/args.batch_size)
  print('steps per epoch: {}'.format(steps_per_epoch))


  if args.lr_decay == 'warmup':
      fn_lr_policy = lambda s: warmup_decay_policy(args.lr, s, args.num_epochs * steps_per_epoch)
  else:
      fn_lr_policy = None

  other_inputs = {}
  os.makedirs(os.path.join(args.output_dir,"runs"),exist_ok=True)
  summary_writer = SummaryWriter(log_dir=os.path.join(args.output_dir,"runs"))
  other_inputs["summary_writer"]=summary_writer

  print('training begins...')
  train(train_data_loader,
        dev_data_loader,
        phone_bilstm,
        optimizer,
        args=args,
        fn_lr_policy=fn_lr_policy,
        device=device,
        other_inputs=other_inputs)

if __name__=="__main__":
    args = parse_args()
    print_dict(vars(args))
    main(args)