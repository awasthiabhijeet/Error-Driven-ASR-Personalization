import torch
import os
import string
import json
from text import _clean_text
from math import floor,ceil

labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

punctuation = string.punctuation
punctuation = punctuation.replace("+", "")
punctuation = punctuation.replace("&", "")
for l in labels:
    punctuation = punctuation.replace(l, "")
table = str.maketrans(punctuation, " " * len(punctuation))

def normalize_string(s, labels=labels, table=table, **unused_kwargs):
    """
    Normalizes string. For example:
    'call me at 8:00 pm!' -> 'call me at eight zero pm'

    Args:
        s: string to normalize
        labels: labels used during model training.

    Returns:
            Normalized string
    """

    def good_token(token, labels):
        s = set(labels)
        for t in token:
            if not t in s:
                return False
        return True

    try:
        text = _clean_text(s, ["english_cleaners"], table).strip()
        return ''.join([t for t in text if good_token(t, labels=labels)])
    except:
        print("WARNING: Normalizing {} failed".format(s))
        return None

def normalized_json_transcript(json_str):
	transcript = json.loads(json_str.strip())["text"]
	return normalize_string(transcript,labels,table)

def print_dict(d):
    maxLen = max([len(ii) for ii in d.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(d.items()):
            print(fmtString % keyPair)

def warmup_decay_policy(initial_lr, step, N, warmup_portion=0.1):
  min_lr = 1e-10
  warmup_steps = floor(warmup_portion*N)
  remaining_steps = N - warmup_steps

  if step <= warmup_steps:
    return initial_lr * (step/warmup_steps)
  else:
    res = initial_lr * ((N-step)/remaining_steps)
    return max(res,min_lr)

def save_model(model, optimizer, epoch, output_dir, save_optimizer=True):
  os.makedirs(output_dir,exist_ok=True)
  class_name = model.__class__.__name__
  file_name = "{0}.pt".format(class_name)
  print("Saving module {0} in {1}".format(class_name, os.path.join(output_dir, file_name)))
  model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
  save_checkpoint={
                  'epoch': epoch,
                  'state_dict': model_to_save.state_dict(),
                  'optimizer': optimizer.state_dict() if save_optimizer else None
                  }

  torch.save(save_checkpoint, os.path.join(output_dir, file_name))
  print('Saved.')