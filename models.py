from __future__ import print_function
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import collections
import math
import copy
torch.manual_seed(1)
np.random.seed(1)


class BIN_Interaction_Flat(nn.Sequential):
    '''
        Interaction Network with 2D interaction map
    '''
    
    def __init__(self, **config):
        super(BIN_Interaction_Flat, self).__init__()
        self.max_d = config['max_drug_seq']
        self.max_p = config['max_protein_seq']
        self.emb_size = config['emb_size']
        self.dropout_rate = config['dropout_rate']
        
        #densenet
        self.scale_down_ratio = config['scale_down_ratio']
        self.growth_rate = config['growth_rate']
        self.transition_rate = config['transition_rate']
        self.num_dense_blocks = config['num_dense_blocks']
        self.kernal_dense_size = config['kernal_dense_size']
        self.batch_size = config['batch_size']
        self.input_dim_drug = config['input_dim_drug']
        self.input_dim_target = config['input_dim_target']
        self.gpus = torch.cuda.device_count()
        self.n_layer = 2
        #encoder
        self.hidden_size = config['emb_size']
        self.intermediate_size = config['intermediate_size']
        self.num_attention_heads = config['num_attention_heads']
        self.attention_probs_dropout_prob = config['attention_probs_dropout_prob']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        
        self.flatten_dim = config['flat_dim'] 
        
        # specialized embedding with positional one
        self.demb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.pemb = Embeddings(self.input_dim_target, self.emb_size, self.max_p, self.dropout_rate)
        
        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size, self.num_attention_heads, self.attention_probs_dropout_prob, self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size, self.num_attention_heads, self.attention_probs_dropout_prob, self.hidden_dropout_prob)
        
        self.icnn = nn.Conv2d(1, 3, 3, padding = 0)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(True),
            
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(True),
            
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            
            #output layer
            nn.Linear(32, 1)
        )
        
    def forward(self, d, p, d_mask, p_mask):
        
        ex_d_mask = d_mask.unsqueeze(1).unsqueeze(2)
        ex_p_mask = p_mask.unsqueeze(1).unsqueeze(2)
        
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0
        
        d_emb = self.demb(d) # batch_size x seq_length x embed_size
        p_emb = self.pemb(p)

        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...
        
        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())
        #print(d_encoded_layers.shape)
        p_encoded_layers = self.p_encoder(p_emb.float(), ex_p_mask.float())
        #print(p_encoded_layers.shape)

        # repeat to have the same tensor size for aggregation   
        d_aug = torch.unsqueeze(d_encoded_layers, 2).repeat(1, 1, self.max_p, 1) # repeat along protein size
        p_aug = torch.unsqueeze(p_encoded_layers, 1).repeat(1, self.max_d, 1, 1) # repeat along drug size
        
        i = d_aug * p_aug # interaction
        i_v = i.view(int(self.batch_size/self.gpus), -1, self.max_d, self.max_p) 
        # batch_size x embed size x max_drug_seq_len x max_protein_seq_len
        i_v = torch.sum(i_v, dim = 1)
        #print(i_v.shape)
        i_v = torch.unsqueeze(i_v, 1)
        #print(i_v.shape)
        
        i_v = F.dropout(i_v, p = self.dropout_rate)        
        
        #f = self.icnn2(self.icnn1(i_v))
        f = self.icnn(i_v)
        
        #print(f.shape)
        
        #f = self.dense_net(f)
        #print(f.shape)
        
        f = f.view(int(self.batch_size/self.gpus), -1)
        #print(f.shape)
        
        #f_encode = torch.cat((d_encoded_layers[:,-1], p_encoded_layers[:,-1]), dim = 1)
        
        #score = self.decoder(torch.cat((f, f_encode), dim = 1))
        score = self.decoder(f)
        return score    
   
# help classes    
    
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    
    
    
class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output    
    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output    

    
class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])    

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            #if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        #if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states