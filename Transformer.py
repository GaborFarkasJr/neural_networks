import torch
from torch import nn
import math

'''
This is a tranformer model that is based on the research paper "Attention is all you need"
'''

# Input size aka d_model

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        '''
        Initialises a Multi-Head Attention Block.
        
        args:
            input_size: Size of the input. Aka. d_model
            num_heads: number of heads for Q, K, V    
        '''
        
        # Input must be dividible by the number of heads
        assert input_size % num_heads == 0
        
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = input_size / num_heads
        
        self.values_linear = nn.Linear(input_size, input_size, bias=False)
        self.key_linear = nn.Linear(input_size, input_size, bias=False)
        self.query_linear = nn.Linear(input_size, input_size, bias=False)
        
        self.output_linear = nn.Linear(input_size, bias=False)
        
    def scaled_dot_product_attention(self, v, k, q, mask=None):
        '''
        Returns the Scaled Dot Product Attention from the v, k, q 
        (Value, Key, Query). Masking can be added optionally 
        (useful for decoders).
        
        Args:
            v: Value Tensor
            k: Key Tensor
            q: Query Tensor
            mask: The masking value (for decoder)
        '''
        
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.input)
        
        # mask for decoding
        if mask != None:
            score = score.masked_fill(mask == 0, -1e9)
        
        # Using the formula from the paper
        softmax_score = torch.softmax(score, -1)
        attention = torch.matmul(softmax_score, v)
        
        return attention
    
    # Shape [batch_size, Sequence_length, Input] -> [batch_size, Sequence_length, Num_heads, Head_dim]
    def split_to_heads(self, x):
        '''
        Splits the input Tensor between the heads and returns it. 
        (shape: [batch_size, Sequence_length, Num_heads, Head_dim])
        
        Args:
            x: Input Tensor (shape: [batch_size, Sequence_length, Input])
        '''
        
        batch_size, sequence_length, _ = x.shape()
        return x.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
    
    # [batch_size, Sequence_length, Num_heads, Head_dim] -> [batch_size, Sequence_length, Input]
    def combine_heads(self, x):
        '''
        Combines the split input Tensor  returns it. 
        (shape: [batch_size, Sequence_length, Input])
        
        Args:
            x: Input Tensor (shape: [batch_size, Sequence_length, Num_heads, Head_dim])
        '''
        
        batch_size, sequence_length, _, _ = x.shape()
        return x.reshape(batch_size, sequence_length, self.input_size)
    
    
    def forward(self, q, k, v, mask=None):
        '''
        Computing the attention.
        
        Args:
            v: Value Tensor
            k: Key Tensor
            q: Query Tensor
            mask: The masking value (for decoder)
        '''
        
        q = self.split_to_heads(self.query_linear(q))
        k = self.split_to_heads(self.query_linear(k))
        v = self.split_to_heads(self.query_linear(v))
        
        attention = self.scaled_dot_product_attention(v, k, q, mask)
        return self.output_linear(self.combine_heads(attention))
        
# Hidden layer size. AKA. d_ff
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeedForward, self).__init__()
        '''
        Creates a Feed Forward block.
        
        Args:
            input_size: Size of the input. Aka d_model
        '''
    
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.relu_activation = nn.ReLU()
        
    # Keep an eye on this for later tweaking
    def forward(self, x):
        '''
        Feed-forward calculation
        
        Args:
            x: Input Tensor
        '''
        
        relu_linear = self.relu_activation(self.linear1(x))
        
        return self.linear2(relu_linear)
    
# Positional Encoding -> for both the encoder and decoder
class PositionalEncoding(nn.Module):
    def __init__(self, input_size, max_sequence):
        super(PositionalEncoding, self).__init__()
        '''
        Positional encding for the inputs. Makes sure that the position of the 
        inputs 
        '''
        
        self.input_size = input_size
        self.max_sequence = max_sequence
        
        even_seq = torch.arange(0, input_size, 2).float()
        # odd_seq = torch.arange(1, input_size, 2).float()
        
        position = torch.arange(0, max_sequence, dtype=torch.float).reshape(max_sequence, 1)
        denom  = torch.pow(10000, (even_seq/input_size))
        self.even_pe = torch.sin(position / denom)
        self.odd_pe = torch.cos(position / denom)
        
        

        

# Encoder Layer

# Decoder Layer

# Transformer model
    