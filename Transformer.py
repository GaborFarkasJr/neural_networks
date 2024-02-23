import torch
from torch import nn
import math

'''
This is a tranformer model that is based on the research paper "Attention is all you need"
'''

# Input size aka d_model -> usually the default is 512

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
    def __init__(self, input_size, max_sequence, dropout):
        super(PositionalEncoding, self).__init__()
        '''
        Positional encding for the inputs. Makes sure that the position of the 
        inputs 

        Args:
            input_size: The size of the input. Aka. d_model
            max_sequence: The maximum sequence length
            dropout: The dropout value
        '''
    
        self.droupout = nn.Dropout(dropout)
        
        even_seq = torch.arange(0, input_size, 2).float()
        # odd_seq = torch.arange(1, input_size, 2).float()
        
        # positions from 0 to max_sequence
        position = torch.arange(0, max_sequence, dtype=torch.float).unsqueeze(1)
        denom  = torch.pow(10000, (even_seq/input_size))

        pe = torch.zeros(max_sequence, input_size)

        # Sin for even values, Cos for odd values
        pe[:, 0::2] = torch.sin(position / denom)
        pe[:, 1::2] = torch.cos(position / denom)

        pe = pe.unsqueeze(0)

        # Some papers add an extra dimension and transpose
        # For simplicity, I just kept it as is

        # Save as state instead of model parameter
        self.register_buffer('pe', pe) 



    def forward(self, x):
        '''
        Adds positional encoding to the inputs

        Args:
            x: Input Tensor [batch, sequence_len, input_size]
        '''
        x += self.pe[:, x.size(1)]

        return self.droupout(x)
        
# Encoder Layer
class Encoder(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, dropout):
        super(Encoder, self).__init__()
        '''
        Encoder block for the Transformer model.

        Args:
            input_size: Size of the input. Aka. d_model
            num_heads: number of heads for Multi-Head Attention block
            hidden_size: Hidden size of inputs. Aka. d_ff
            dropout: The dropout value
        '''

        # Embedding needed as well

        # Possibly some normalization

        # Dropout is added to Positional encoding
        self.mutli_head_attention_block = MultiHeadAttention(input_size, num_heads)
        self.feed_forward_block = FeedForward(input_size, hidden_size)
        self.normalisation_layer1 = nn.LayerNorm(input_size)
        self.normalisation_layer2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Encodes the input for the deoder block.

        Args:
            input_size: The size of the input. Aka. d_model
            num_heads: The number of heads
            hidden_size: The size of the hidden layer. Aka. d_ff.
        '''

        attention = self.dropout(self.mutli_head_attention_block(x, x, x))
        normalised_attiention = self.normalisation_layer1(x + attention)

        feed_forward = self.dropout(self.feed_forward_block)
        output = self.normalisation_layer2(feed_forward + normalised_attiention)

        return output


# Decoder Layer
class Decoder(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, dropout):
        super(Decoder, self).__init__()
        '''
        Decoder block for the Transformer model.

        Args:
            input_size: Size of the input. Aka. d_model
            num_heads: number of heads for Multi-Head Attention block
            hidden_size: Hidden size of inputs. Aka. d_ff
            dropout: The dropout value
        '''

        self.masked_mutli_head_attention_block = MultiHeadAttention(input_size, num_heads)
        self.mutli_head_attention_block = MultiHeadAttention(input_size, num_heads)
        self.feed_forward_block = FeedForward(input_size, hidden_size)

        self.normalisation_layer1 = nn.LayerNorm(input_size)
        self.normalisation_layer2 = nn.LayerNorm(input_size)
        self.normalisation_layer3 = nn.LayerNorm(input_size)

        self.dropout = nn.Dropout(dropout)



    def forward(self, x, encoder_input, mask):
        '''
        Decodes the input from the Encoder for the output.

        Args:
            input_size: The size of the input. Aka. d_model
            num_heads: The number of heads
            hidden_size: The size of the hidden layer. Aka. d_ff.
        '''

        masked_attention = self.dropout(self.masked_mutli_head_attention_block(x, x, x, mask))
        normalised_masked_attention = self.normalisation_layer1(x + masked_attention)

        attention = self.dropout(self.mutli_head_attention_block(encoder_input, encoder_input, x))
        normalised_attention = self.normalisation_layer2(normalised_masked_attention + attention)

        feed_forward = self.dropout(self.feed_forward_block(normalised_attention))
        output = self.normalisation_layer3(normalised_attention + feed_forward)

        return output
    
# Transformer model
class SLRTransformer(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers,
                 max_sequence,
                 encoder_vocab_size, 
                 decoder_vocab_size, 
                 num_heads, 
                 dropout
                 ):
        super(SLRTransformer, self).__init__()
        '''
        SLR Transformer that has been tweaked slightly to allow for the classification if signs.
        The output will not be fead through a softmax layer, since it will make use of 
        cross-entropy loss (which automatically implements this). The source vocab size and the
        tarket vocab size are kept separate to allow for flexibility

        Args:
            input_size: Size of the input. Aka. d_model
            hidden_size: The size of the hidden layer. Aka. d_ff.
            num_layers: Number of encoder & decoder layers
            max_sequence: The maximum sequence length
            input_vocab_size: Vocabulary size of the input
            output_vocab_size: Vocabulary size of the output
            num_heads: The number of heads
            dropout: The dropout value
        '''

        self.max_sequence = max_sequence


        self.encoder_embedding = nn.Embedding(encoder_vocab_size, input_size)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, input_size)

        self.encoder_pos_encoding = PositionalEncoding(input_size, max_sequence, dropout)

        self.encoder_block = nn.ModuleList([Encoder(input_size, num_heads, hidden_size, dropout) for _ in range(num_layers)])
        self.decoder_block = nn.ModuleList([Decoder(input_size, num_heads, hidden_size, dropout) for _ in range(num_layers)])

        self.linear_output = (input_size, decoder_vocab_size)
        self.droput = nn.Dropout(dropout)
    
    def generate_mask(self):
