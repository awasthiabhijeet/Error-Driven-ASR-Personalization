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
import pickle
import json

from model import ErrorClassifierPhoneBiLSTM, ErrorClassifierPhoneBiLSTM_V2
from data import inference_collate_fn, load_phoneme_sequences, phone_list
from metrics import xent_loss, error_classifier_errors, get_precision_recall_f1
from helpers import print_dict, warmup_decay_policy, save_model

def eval(eval_data_loader,model,args,device):
  with torch.no_grad():
    pj_weights = []

    for data in eval_data_loader:
      data = [item.to(device) for item in data]
      phonemes, padding_positions, sequence_lengths = data
      one_hot_phones = F.one_hot(phonemes,num_classes=len(phone_list)).float()
      model.eval()
      logits = model(phonemes,sequence_lengths)
      probs = F.softmax(logits,-1)
      error_probs = probs[:,:,1]
      error_probs = error_probs * padding_positions
      pj_error_probs = torch.unsqueeze(error_probs,-1) * one_hot_phones
      pj_error_probs = torch.sum(pj_error_probs,1)
      batch_pj_wts = pj_error_probs.cpu().numpy().tolist()
      pj_weights.extend(batch_pj_wts)
  return pj_weights

def parse_args():
    parser = argparse.ArgumentParser(description='infer error model')
    parser.add_argument("--batch_size", default=64, type=int, help='data batch size')
    parser.add_argument("--json_path", type=str, required=True, help='path to json file')
    parser.add_argument("--seed", default=42, type=int, help='seed')
    parser.add_argument("--pretrained_ckpt", default=None, type=str, help='path to pretrained ckpt')
    parser.add_argument("--input_size",default=64,type=int,help='size of phone embeddings')
    parser.add_argument("--hidden_size",default=64,type=int,help='size of hidden cells of lstm')
    parser.add_argument("--num_layers",default=4,type=int,help='number of lstm layers')
    parser.add_argument("--output_dir",default=None,type=str,help='path to dump weights')
    args=parser.parse_args()
    return args

def main(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  json_path = args.json_path

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  print('loading data....')
  data = load_phoneme_sequences([json_path],remove_duplicates=False)

  print('data_size: {}'.format(len(data)))

  print('creating data loaders...')

  data_loader = torch.utils.data.DataLoader(data,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            collate_fn=inference_collate_fn,
                                            drop_last=False)

  print('creating model...')
  phone_bilstm = ErrorClassifierPhoneBiLSTM_V2(input_size=args.input_size,hidden_size=args.hidden_size,num_layers=args.num_layers)
  if args.pretrained_ckpt:
    print('loading pretrained cpkt from : {}'.format(args.pretrained_ckpt))
    checkpoint = torch.load(args.pretrained_ckpt, map_location="cpu")
    #del checkpoint['state_dict']['fc_layer_2.weight'] 
    #del checkpoint['state_dict']['fc_layer_2.bias']
    phone_bilstm.load_state_dict(checkpoint['state_dict'],strict=True)
    
  phone_bilstm.to(device)


  print('inference begins...')
  pj_weights = eval(data_loader,phone_bilstm,args,device)
  pretrained_ckpt_dir = os.path.split(args.pretrained_ckpt)[0]
  if args.output_dir is None:
    args.output_dir = pretrained_ckpt_dir

  pj_weights_path = os.path.join(args.output_dir,'weights.pkl')
  pickle.dump(pj_weights, open(pj_weights_path,"wb"))

if __name__=="__main__":
    args = parse_args()
    print_dict(vars(args))
    main(args)