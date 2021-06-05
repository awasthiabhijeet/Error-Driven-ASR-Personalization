import torch
import torch.nn as nn
import torch.nn.functional as F

xent = nn.CrossEntropyLoss(reduction='none')

def xent_loss(logits, error_positions, padding_positions, copy_weight=1.0):
  logits = torch.transpose(logits,1,2) # (B x C x T)
  loss = xent(logits, error_positions) # (B x T)
  loss = loss * padding_positions
  #loss = copy_weight * loss * (padding_positions==0).float() + loss * (padding_positions==1.0).float()
  loss = torch.sum(loss,-1)
  loss = torch.mean(loss)
  return loss

def error_classifier_errors(predictions, error_positions, padding_positions):
  error = (predictions != error_positions).float() * padding_positions
  error = torch.sum(error)
  true_positives = (predictions == 1.0).float() * (error_positions == 1.0).float() * padding_positions
  true_positives = torch.sum(true_positives)
  false_positives = (predictions == 1.0).float() * (error_positions == 0.0).float() * padding_positions
  false_positives = torch.sum(false_positives)
  false_negatives = (predictions == 0.0).float() * (error_positions == 1.0).float() * padding_positions
  false_negatives = torch.sum(false_negatives)

  return error, true_positives, false_positives, false_negatives

def get_precision_recall_f1(true_positives, false_positives, false_negatives):
  precision = (true_positives)/(true_positives + false_positives + 1e-10)
  recall = (true_positives)/(true_positives + false_negatives + 1e-10)
  f1 = (2*precision*recall)/(precision + recall + 1e-10)

  return precision*100, recall*100, f1*100