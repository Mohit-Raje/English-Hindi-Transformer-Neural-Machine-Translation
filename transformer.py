import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sentencepiece as spm
import math
import torch.nn.functional as F
from torch.utils.data import Dataset , DataLoader
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2



## Embedding Layer
class TokenEmbedding(nn.Module):

  def __init__(self , vocab_size , embed_size):

    super().__init__()
    self.Embedding = nn.Embedding(vocab_size , embed_size)
    self.embed_size = embed_size

  def forward(self , x):
      #scaling - need ?
      ## Initailly the embedding values are randomly assigned , which are small numbers so they have low variance and if such value are
      #passed to the transformer , then then vanish gradient problem could occur , to avoid this issue embeddings are scaled .

      scaled_embedding = self.Embedding(x) * torch.sqrt(torch.tensor(self.embed_size , dtype=torch.float32))
      return scaled_embedding

##Positional Encoding 
class PositionalEncoding(nn.Module):

    def __init__(self , d_model , max_len=5000):

      super().__init__()

      pe = torch.zeros(max_len , d_model)
      position = torch.arange(0 , max_len , dtype=torch.float).unsqueeze(1)

      div_term = torch.exp(torch.arange(0 , d_model , 2).float() * (-math.log(10000.0)/d_model))

      pe[: , 0::2] = torch.sin(position * div_term)
      pe[: , 1::2] = torch.cos(position * div_term)

      pe = pe.unsqueeze(0)


      self.register_buffer('pe' , pe)

    def forward(self , x):

      seq_len = x.size(1)
      x = x + self.pe[: , :seq_len]
      return x


##Encoder :


##Multihead Attention
class MultiHeadAttention(nn.Module):

  def __init__(self , d_model , num_heads):

    super().__init__()

    assert d_model % num_heads == 0

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = self.d_model // self.num_heads

    self.W_Q = nn.Linear(d_model , d_model)
    self.W_K = nn.Linear(d_model , d_model)
    self.W_V = nn.Linear(d_model , d_model)

    self.W_O = nn.Linear(d_model , d_model)

    self.scale = math.sqrt(self.d_k)

  def forward(self , x , mask=None):

    batch_size , seq_len , _ = x.size() #x.size() == torch.Size([32, 10, 512])

    Q = self.W_Q(x)
    K = self.W_K(x)
    V = self.W_V(x)

    Q = Q.view(batch_size , seq_len , self.num_heads , self.d_k).transpose(1,2)
    K = K.view(batch_size , seq_len , self.num_heads , self.d_k).transpose(1,2)
    V = V.view(batch_size , seq_len , self.num_heads , self.d_k).transpose(1,2)

    scores = torch.matmul(Q , K.transpose(-2 , -1))/self.scale

    if mask is not None:
      if mask.dim() == 2:
          mask = mask.unsqueeze(1).unsqueeze(1)
      if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

      scores = scores.masked_fill(~mask, -1e9)

    attention_weights = F.softmax(scores , dim=-1)

    attention_output = torch.matmul(attention_weights , V) #(batch , head , seq ,d_k)

    attention_output = attention_output.transpose(1,2).contiguous().view(batch_size , seq_len , self.d_model)

    output = self.W_O(attention_output)

    return output , attention_weights

## Add and Norm Layer
class AddNorm(nn.Module):

  def __init__(self , d_model , eps=1e-6):

    super().__init__()
    self.norm = nn.LayerNorm(d_model , eps=eps)

  def forward(self , x , sublayer_output):
    return self.norm(x + sublayer_output)

##Feed Forward Neural Network
class FFNN(nn.Module):

  def __init__(self , d_models , d_ff):

    super().__init__()
    self.L1 = nn.Linear(d_models , d_ff)
    self.L2 = nn.Linear(d_ff , d_models)

  def forward(self , x):

    return self.L2(F.relu(self.L1(x)))


## Encoder Layer
class EncoderLayer(nn.Module):

  def __init__(self , d_model , num_heads , d_ff):

    super().__init__()

    self.mha = MultiHeadAttention(d_model , num_heads)
    self.add_norm1 = AddNorm(d_model)

    self.ffnn = FFNN(d_model , d_ff)
    self.add_norm2 = AddNorm(d_model)


  def forward(self , x , mask=None):

    context_vector , _ = self.mha(x)
    x = self.add_norm1(x , context_vector)

    ffnn_output = self.ffnn(x)
    final_result = self.add_norm2(x  , ffnn_output)

    return final_result

## Complete Encoder Block
class Encoder(nn.Module):

  def __init__(self , vocab_size , d_model , num_heads , num_layers , d_ff , max_len=512):

    super().__init__()

    self.embedding = TokenEmbedding(vocab_size , d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_len = max_len)


    self.layers = nn.ModuleList([
        EncoderLayer(d_model , num_heads ,d_ff)
        for _ in range(num_layers)
    ])

    self.norm = nn.LayerNorm(d_model)

  def forward(self , x , mask=None):

    x = self.embedding(x)
    x = self.pos_encoding(x)

    for layer in self.layers:
      x = layer(x , mask)

    return self.norm(x)




## Decoder Block:

##Masked Multi 
class MaskedMultiHeadAttention(nn.Module):


  def __init__(self , d_model , num_heads):

    super().__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads


    self.W_Q = nn.Linear(d_model , d_model)
    self.W_K = nn.Linear(d_model , d_model)
    self.W_V = nn.Linear(d_model , d_model)

    self.W_O = nn.Linear(d_model , d_model)
    self.scale = math.sqrt(self.d_k)  ## Dim of key vector per head

  def forward(self , x , look_ahead_mask=None):

    batch_size , seq_len , _ = x.size()

    Q = self.W_Q(x)
    K = self.W_K(x)
    V = self.W_V(x)

    Q = Q.view(batch_size , seq_len , self.num_heads , self.d_k).transpose(1,2)
    K = K.view(batch_size , seq_len ,self.num_heads  , self.d_k).transpose(1,2)
    V = V.view(batch_size , seq_len , self.num_heads , self.d_k).transpose(1,2)

    scores = torch.matmul(Q , K.transpose(-2 , -1))/self.scale

    if look_ahead_mask is not None:
      if look_ahead_mask.dim() == 2:
          look_ahead_mask = look_ahead_mask.unsqueeze(1).unsqueeze(1)

      if look_ahead_mask.dtype != torch.bool:
        look_ahead_mask = look_ahead_mask.to(torch.bool)

      scores = scores.masked_fill(~look_ahead_mask, -1e9)

    atten_weight = F.softmax(scores  , dim=-1)

    atten_output = torch.matmul(atten_weight , V)

    atten_output = atten_output.transpose(1 , 2).contiguous().view(batch_size , seq_len , self.d_model)

    output = self.W_O(atten_output)

    return output , atten_weight


##Cross Attention:

class Crossattention(nn.Module):

  def __init__(self , d_model , num_heads):

    super().__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model//num_heads

    self.W_Q = nn.Linear(d_model , d_model)
    self.W_K = nn.Linear(d_model , d_model)
    self.W_V = nn.Linear(d_model , d_model)

    self.W_O = nn.Linear(d_model , d_model)
    self.scale = math.sqrt(self.d_k)

  def forward(self , query , key , value , padding_mask = None):

    batch_size = query.size(0)  ##query -> Decoder
    target_seq_len = query.size(1)
    src_seq_len = key.size(1)

    Q = self.W_Q(query)
    K = self.W_K(key)
    V = self.W_V(value)


    Q = Q.view(batch_size , target_seq_len , self.num_heads , self.d_k).transpose(1 , 2)
    K = K.view(batch_size , src_seq_len , self.num_heads , self.d_k).transpose(1 , 2)
    V = V.view(batch_size , src_seq_len , self.num_heads , self.d_k).transpose(1 , 2)

    scores = torch.matmul(Q , K.transpose(-2 , -1))/self.scale


    if padding_mask is not None:
      if padding_mask.dim() == 2:
          padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)

      if padding_mask.dtype != torch.bool:
        padding_mask = padding_mask.to(torch.bool)

      scores = scores.masked_fill(~padding_mask, -1e9)


    attention_weights = F.softmax(scores , dim=-1)

    attention_output = torch.matmul(attention_weights , V)

    attention_output = attention_output.transpose(1 , 2).contiguous().view(batch_size , target_seq_len , self.d_model)

    output = self.W_O(attention_output)

    return output , attention_weights


class DecoderLayer(nn.Module):

  def __init__(self , d_model , num_heads , d_ff):

    super().__init__()

    self.masked_attention = MaskedMultiHeadAttention(d_model , num_heads)
    self.add_norm1 = AddNorm(d_model)

    self.cross_attention = Crossattention(d_model , num_heads)
    self.add_norm2 = AddNorm(d_model)

    self.ffnn = FFNN(d_model , d_ff)
    self.add_norm3 = AddNorm(d_model)


  def forward(self , x , enc_output , look_ahead_mask=None , padded_mask=None):

    masked_attention_output , _ = self.masked_attention(x , look_ahead_mask)
    x = self.add_norm1(x, masked_attention_output)

    cross_attention_output , _  = self.cross_attention(x , enc_output , enc_output , padded_mask)
    x = self.add_norm2(x , cross_attention_output)

    ffnn_output = self.ffnn(x)
    x = self.add_norm3(x , ffnn_output)

    return x



class Decoder(nn.Module):

  def __init__(self , vocab_size , d_model , num_heads , num_layers , d_ff , max_len=512):

    super().__init__()

    self.embedding = TokenEmbedding(vocab_size , d_model)
    self.pos_embedding = PositionalEncoding(d_model , max_len = max_len)

    self.layers = nn.ModuleList([
          DecoderLayer(d_model ,num_heads , d_ff)
          for _ in range(num_layers)
    ])

    self.norm = nn.LayerNorm(d_model)

  def forward(self , x , enc_output , look_ahead_mask = None , padding_mask=None):

    x = self.embedding(x)
    x = self.pos_embedding(x)

    for layer in self.layers:
      x = layer(x , enc_output , look_ahead_mask , padding_mask)

    return self.norm(x)


def create_padding_mask(seq, pad_index=0):
    return (seq != pad_index).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_ids, tgt_ids, src_padding_mask=None, tgt_look_ahead_mask=None, tgt_padding_mask=None):
        # src_ids: (batch, src_seq)  — Encoder expects token ids, since it embeds internally
        # tgt_ids: (batch, tgt_seq)  — Decoder expects token ids, since it embeds internally

        # If user didn't pass padding masks, create them
        if src_padding_mask is None:
            src_padding_mask = create_padding_mask(src_ids)   # (batch,1,1,src_seq)
        if tgt_padding_mask is None:
            tgt_padding_mask = create_padding_mask(tgt_ids)

        if tgt_look_ahead_mask is None:
            tgt_look_ahead_mask = create_look_ahead_mask(tgt_ids.size(1)).to(src_ids.device)

        # Now call encoder and decoder with correct order
        enc_out = self.encoder(src_ids, src_padding_mask)  # encoder will embed internally
        dec_out = self.decoder(tgt_ids, enc_out, tgt_look_ahead_mask, src_padding_mask)

        logits = self.fc_out(dec_out)  # (batch, tgt_seq, vocab)
        return logits


