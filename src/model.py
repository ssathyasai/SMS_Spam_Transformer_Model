"""
Spam Transformer Model
Modified Transformer architecture for binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions
    Injects information about position of tokens
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    Allows model to focus on different parts of the sequence
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention
        Attention(Q,K,V) = softmax(QK^T/√d_k)V
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        # query, key, value: [seq_len, batch_size, d_model]
        seq_len_q, batch_size, _ = query.size()
        seq_len_k = key.size(0)

        Q = self.W_q(query).view(seq_len_q, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        K = self.W_k(key).view(seq_len_k, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        V = self.W_v(value).view(seq_len_k, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)

        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        output = self.W_o(attn_output).transpose(0, 1)

        return output

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    Consists of Multi-Head Attention and Feed-Forward with residual connections
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer
    Consists of Masked Self-Attention, Cross-Attention, and Feed-Forward
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with encoder output
        attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class SpamTransformer(nn.Module):
    """
    Modified Transformer model for SMS Spam Detection
    Features:
    - Memory mechanism for classification
    - Binary classification output
    - Positional encoding for sequence information
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, d_ff=2048, dropout=0.1, memory_length=10):
        super(SpamTransformer, self).__init__()
        
        self.d_model = d_model
        self.memory_length = memory_length
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Memory (trainable parameters) - replaces target sequence in classification
        self.memory = nn.Parameter(torch.randn(memory_length, 1, d_model) * 0.1)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Final classification layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.fc3 = nn.Linear(d_model // 4, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, src_mask=None):
        """
        Forward pass through the model
        
        Steps:
        1. Embed input tokens and add positional encoding
        2. Pass through encoder layers (self-attention)
        3. Prepare memory for decoder
        4. Pass through decoder layers with memory
        5. Aggregate decoder outputs and classify
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Step 1: Embedding and positional encoding
        # x: [batch_size, seq_len] -> [seq_len, batch_size, d_model]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        
        # Step 2: Encoder layers
        # Process input sequence through multiple encoder layers
        encoder_output = x
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, src_mask)
        
        # Step 3: Prepare decoder input (memory)
        # Expand memory for batch processing
        memory = self.memory.repeat(1, batch_size, 1)
        
        # Step 4: Decoder layers
        # Process memory with encoder output using cross-attention
        decoder_output = memory
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, src_mask)
        
        # Step 5: Classification
        # Aggregate decoder outputs (mean pooling)
        aggregated = decoder_output.mean(dim=0)  # [batch_size, d_model]
        
        # Through classification layers
        out = self.dropout(F.relu(self.fc1(aggregated)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.fc3(out)
        
        # Final sigmoid for binary classification
        output = self.sigmoid(out).squeeze(-1)
        
        return output

class SpamTransformerWithEmbeddings(SpamTransformer):
    """
    Extended version that can handle pre-trained embeddings
    """
    def __init__(self, vocab_size, embedding_weights=None, **kwargs):
        super(SpamTransformerWithEmbeddings, self).__init__(vocab_size, **kwargs)
        
        if embedding_weights is not None:
            # Load pre-trained embeddings (GloVe)
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)