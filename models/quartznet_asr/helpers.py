# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from enum import Enum
from metrics import word_error_rate, f_wer, f_cer
import json

import multiprocessing
cpu_count = multiprocessing.cpu_count()

import numpy as np

import pdb



AmpOptimizations = ["O0", "O1", "O2", "O3"]

def print_once(msg):
    if (not torch.distributed.is_initialized() or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):
        print(msg)

def add_ctc_labels(labels):
    if not isinstance(labels, list):
        raise ValueError("labels must be a list of symbols")
    labels.append("<BLANK>")
    return labels

def __ctc_decoder_predictions_tensor(tensor, labels):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Returns prediction
    Args:
        tensor: model output tensor
        label: A list of labels
    Returns:
        prediction
    """
    blank_id = len(labels) - 1
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    prediction_cpu_tensor = tensor.long().cpu()
    # iterate over batch
    for ind in range(prediction_cpu_tensor.shape[0]):
        prediction = prediction_cpu_tensor[ind].numpy().tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = len(labels) - 1 # id of a blank symbol
        for p in prediction:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = ''.join([labels_map[c] for c in decoded_prediction])
        hypotheses.append(hypothesis)
    return hypotheses


def monitor_asr_train_progress(tensors: list, labels: list, do_print=True):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Prints wer and prediction examples to screen
    Args:
        tensors: A list of 3 tensors (predictions, targets, target_lengths)
        labels: A list of labels

    Returns:
        word error rate
    """
    references = []

    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    with torch.no_grad():
        targets_cpu_tensor = tensors[1].long().cpu()
        tgt_lenths_cpu_tensor = tensors[2].long().cpu()

        # iterate over batch
        for ind in range(targets_cpu_tensor.shape[0]):
            tgt_len = tgt_lenths_cpu_tensor[ind].item()
            target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
            reference = ''.join([labels_map[c] for c in target])
            references.append(reference)
        hypotheses = __ctc_decoder_predictions_tensor(tensors[0], labels=labels)
    tag = "training_batch_WER"
    wer,wer_list, _, _ = word_error_rate(hypotheses, references)

    if do_print:
        print_once('{0}: {1}'.format(tag, wer))
        print_once('Prediction: {0}'.format(hypotheses[0]))
        print_once('Reference: {0}'.format(references[0]))
        
    return wer,wer_list


def __gather_losses(losses_list: list) -> list:
    return [torch.mean(torch.stack(losses_list))]


def __gather_predictions(predictions_list: list, labels: list) -> list:
    results = []
    for prediction in predictions_list:
        results += __ctc_decoder_predictions_tensor(prediction, labels=labels)
    return results


def __gather_transcripts(transcript_list: list, transcript_len_list: list,
                                                 labels: list) -> list:
    results = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    # iterate over workers
    for t, ln in zip(transcript_list, transcript_len_list):
        # iterate over batch
        t_lc = t.long().cpu()
        ln_lc = ln.long().cpu()
        for ind in range(t.shape[0]):
            tgt_len = ln_lc[ind].item()
            target = t_lc[ind][:tgt_len].numpy().tolist()
            reference = ''.join([labels_map[c] for c in target])
            results.append(reference)
    return results

def convert_to_strings(out, seq_len,vocab):
    results = []
    for b, batch in enumerate(out):
        utterances = []
        for p, utt in enumerate(batch):
            size = seq_len[b][p]
            if size > 0:
                transcript = ''.join(map(lambda x: vocab[x.item()], utt[0:size]))
            else:
                transcript = ''
            utterances.append(transcript)
        results.append(utterances)
    return results

def process_evaluation_batch(tensors: dict, global_vars: dict, labels: list):
    """
    Processes results of an iteration and saves it in global_vars
    Args:
        tensors: dictionary with results of an evaluation iteration, e.g. loss, predictions, transcript, and output
        global_vars: dictionary where processes results of iteration are saved
        labels: A list of labels
    """
    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['EvalLoss'] += __gather_losses(v)
        elif kv.startswith('predictions'):
            global_vars['predictions'] += __gather_predictions(v, labels=labels)
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v
        elif kv.startswith('output'):
            logits_list = v
            global_vars['logits'] += v
        elif kv.startswith('encoded_length'):
            encoded_len_list = v
            global_vars['encoded_lens'] += v

    character_transcripts = __gather_transcripts(transcript_list,
                                           transcript_len_list,
                                           labels=labels)
    global_vars['transcripts'] += character_transcripts


def process_evaluation_epoch(global_vars: dict, tag=None):
    """
    Processes results from each worker at the end of evaluation and combine to final result
    Args:
        global_vars: dictionary containing information of entire evaluation
    Return:
        wer: final word error rate
        loss: final loss
    """
    if 'EvalLoss' in global_vars:
        eloss = torch.mean(torch.stack(global_vars['EvalLoss'])).item()
    else:
        eloss = None
    hypotheses = global_vars['predictions']
    references = global_vars['transcripts']

    wer, wer_list, scores, num_words = word_error_rate(hypotheses=hypotheses, references=references)
    cer, _,_,_ = word_error_rate(hypotheses=hypotheses, references=references, use_cer=True)
    multi_gpu = torch.distributed.is_initialized()
    if multi_gpu:
        if eloss is not None:
            eloss /= torch.distributed.get_world_size()
            eloss_tensor = torch.tensor(eloss).cuda()
            dist.all_reduce(eloss_tensor)
            eloss = eloss_tensor.item()
            del eloss_tensor

        scores_tensor = torch.tensor(scores).cuda()
        dist.all_reduce(scores_tensor)
        scores = scores_tensor.item()
        del scores_tensor
        num_words_tensor = torch.tensor(num_words).cuda()
        dist.all_reduce(num_words_tensor)
        num_words = num_words_tensor.item()
        del num_words_tensor
        wer = scores *1.0/num_words
    return wer, cer, eloss



def norm(x):
    if not isinstance(x, list):
        if not isinstance(x, tuple):
            return x
    return x[0]


def print_dict(d):
    maxLen = max([len(ii) for ii in d.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(d.items()):
            print(fmtString % keyPair)



def model_multi_gpu(model, multi_gpu=False):
    if multi_gpu:
        model = DDP(model)
        print('DDP(model)')
    return model

def print_sentence_wise_wer(hypotheses, references,output_file,input_file):
    wav_filenames = []
    with open(input_file, "r", encoding="utf-8") as f:
        
        #json_out = json.load(f)
        #wav_filenames = [item['files'][0]['fname'] for item in json_out]
        
        wav_filenames = [json.loads(line.strip())["audio_filepath"] for line in f]
        
        '''
        for line in f:
            line = line.strip()
            line = ast.literal_eval(line)
            wav_filenames.append(line['audio_filepath'])
        '''    
        
    assert len(hypotheses) == len(references)
    assert len(hypotheses) == len(wav_filenames)

    wers = []
    cers = []

    for hyp,ref in zip(hypotheses,references):
        wers.append(f_wer(hyp,ref))
        cers.append(f_cer(hyp,ref))
    
    #wer_cer_diff = [item1-item2 for item1,item2 in zip(wers,cers)]
    #merged_list = zip(hypotheses,references,wers,cers,wav_filenames,wer_cer_diff)
    #hypotheses,references,wers,cers,wav_filenames, wer_cer_diff = list(zip(*sorted(merged_list,key=itemgetter(2),reverse=False)))
    with open(output_file,'w') as f:
        for hyp,ref,wer,cer,wav_filename in zip(hypotheses,references,wers,cers,wav_filenames):
            f.write(wav_filename+"\n")
            f.write("WER: "+str(wer)+'\n')
            f.write("CER: "+ str(cer) +'\n')
            f.write("Ref: "+ref+'\n')
            f.write("Hyp: "+hyp+'\n')
            f.write('\n')
    return wav_filenames

def construct_padding(seq_lens, max_len=None, padding_value=1):
    B = seq_lens.shape[0]
    if max_len is None:
        max_len = torch.max(seq_lens)
    padding = torch.zeros([B,max_len])
    for i,s_len in enumerate(seq_lens):
        padding[i][0:s_len]=padding_value
    return padding


def make_bn_layers_in_eval_mode(jasper_encoder):
    from quartznet_model import JasperBlock
    for item in jasper_encoder.encoder:
        assert isinstance(item, JasperBlock)
        for module in item.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()