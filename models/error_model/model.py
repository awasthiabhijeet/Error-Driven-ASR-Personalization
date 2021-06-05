import torch
import torch.nn as nn
import torch.nn.functional as F

from data import phone_list, Phonemes, coarse_phone_to_fine_phone

class ErrorClassifierPhoneBiLSTM(nn.Module):
  def __init__(self, input_size=512, hidden_size=256, vocab_size=len(phone_list), num_layers=4, output_size=2):
    super(ErrorClassifierPhoneBiLSTM, self).__init__()
    
    self.phone_embeddings = nn.Embedding(vocab_size,input_size) 
    self.lstm = nn.LSTM(input_size,
                  hidden_size,
                  num_layers,
                  batch_first=True,
                  dropout=0.1,
                  bidirectional=True)
    self.fc_layer_1 = nn.Linear(2*hidden_size, input_size)
    self.fc_layer_2 = nn.Linear(input_size,output_size)

  def forward(self,inputs,input_lengths):
    #input: (B x T)
    inputs = self.phone_embeddings(inputs) # (B x T x D)
    inputs = nn.utils.rnn.pack_padded_sequence(inputs,input_lengths,batch_first=True,enforce_sorted=False)
    lstm_output, hidden = self.lstm(inputs)
    lstm_output,_ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
    lstm_output = F.relu(lstm_output)
    lstm_output = self.fc_layer_1(lstm_output)
    #lstm_output = F.relu(lstm_output)
    logits = self.fc_layer_2(lstm_output)
    return logits


class ErrorClassifierPhoneBiLSTM_V2(nn.Module):
  def __init__(self, input_size=512, hidden_size=256, vocab_size=len(phone_list), num_layers=4, output_size=2):
    super(ErrorClassifierPhoneBiLSTM_V2, self).__init__()
    VOWEL_EMBED_SIZE = 8
    FINE_EMBED_SIZE = 8
    self.vocab_size = vocab_size
    self.phone_embeddings = nn.Embedding(vocab_size,input_size)
    self.vowel_embeddings = nn.Embedding(2,VOWEL_EMBED_SIZE)
    self.fine_embeddings = nn.Embedding(10,FINE_EMBED_SIZE)
    self.vowel_mask = [1 if item in Phonemes.vowels else 0 for item in phone_list]
    self.fine_mask = [coarse_phone_to_fine_phone(phone) for phone in phone_list]

    self.lstm = nn.LSTM(input_size + VOWEL_EMBED_SIZE + FINE_EMBED_SIZE,
                  hidden_size,
                  num_layers,
                  batch_first=True,
                  dropout=0.1,
                  bidirectional=True)
    self.fc_layer_1 = nn.Linear(2*hidden_size, 2*hidden_size)
    self.fc_layer_2 = nn.Linear(2*hidden_size,output_size)

  def forward(self,inputs,input_lengths):
    #input: (B x T)
    ph_vowel_embeddings = self.vowel_embeddings(torch.tensor(self.vowel_mask,dtype=torch.int64,device=inputs.device)) # (V x 8)
    ph_vowel_embeddings = torch.unsqueeze(ph_vowel_embeddings,0) # (1 x V x 8)
    
    ph_fine_embeddings = self.fine_embeddings(torch.tensor(self.fine_mask,dtype=torch.int64,device=inputs.device)) # (V x 24)
    ph_fine_embeddings = torch.unsqueeze(ph_fine_embeddings,0) # (1 x V x 24)
    
    one_hot_inputs = F.one_hot(inputs,self.vocab_size).float() # (B x T x V)
    vowel_inputs = torch.matmul(one_hot_inputs, ph_vowel_embeddings)
    fine_inputs = torch.matmul(one_hot_inputs, ph_fine_embeddings)
    inputs = self.phone_embeddings(inputs) # (B x T x D)
    inputs = torch.cat([vowel_inputs, fine_inputs, inputs],-1)


    inputs = nn.utils.rnn.pack_padded_sequence(inputs,input_lengths,batch_first=True,enforce_sorted=False)
    lstm_output, hidden = self.lstm(inputs)
    lstm_output,_ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
    lstm_output = self.fc_layer_1(lstm_output)
    lstm_output = F.dropout(lstm_output,0.1)
    lstm_output = F.relu(lstm_output)
    #lstm_output = F.relu(lstm_output)
    logits = self.fc_layer_2(lstm_output)
    return logits

