import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class PositionalEncoding(nn.Module):
    # Step in architecture: Positional Encoding - injects position info using sine/cosine [file:1]
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class SpamTransformer(nn.Module):
    # Complete architecture from paper: Encoder-Decoder with Memory [file:1]
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4, 
                 dim_feedforward=512, memory_len=32, dropout=0.1, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(vocab_size, d_model)  # Input Embeddings
        self.pos_encoder_msg = PositionalEncoding(d_model, max_len)  # Positional Encoding for input messages
        self.memory = nn.Parameter(torch.randn(memory_len, 1, d_model))  # Trainable Memory (substitute for target seq [file:1])
        self.pos_encoder_mem = PositionalEncoding(d_model, memory_len)  # Positional Encoding for Memory
        
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers)  # Encoder Stack: Self-Attention + Linear FFN
        
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers)  # Decoder Stack: Self-Attn on Memory + Enc-Dec Attn + Linear FFN
        
        self.linear = nn.Linear(d_model, 1)  # Linear layers after decoder
        self.sigmoid = nn.Sigmoid()  # Final Activation for binary classification [file:1]

    def forward(self, src, src_mask=None, memory_mask=None):
        # src: (batch, seq_len) token indices
        src_emb = self.embeddings(src) * math.sqrt(self.d_model)  # Embeddings step
        src_emb = self.pos_encoder_msg(src_emb.transpose(0,1)).transpose(0,1)  # Positional Encoding on input
        
        memory_emb = self.pos_encoder_mem(self.memory).transpose(0,1)  # Positional Encoding on Memory
        
        # Encoder: Self-Attention on input sequence
        encoder_out = self.encoder(src_emb, src_key_padding_mask=src_mask)
        
        # Decoder: Masked Self-Attn on Memory + Encoder-Decoder Attention
        decoder_out = self.decoder(memory_emb, encoder_out, tgt_key_padding_mask=memory_mask)
        
        # Global average pool decoder output
        decoder_out = decoder_out.mean(dim=0)
        output = self.sigmoid(self.linear(decoder_out)).squeeze(-1)  # Linear + Sigmoid Activation
        return output
