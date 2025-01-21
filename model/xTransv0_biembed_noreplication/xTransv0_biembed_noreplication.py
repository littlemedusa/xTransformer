import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import weight_norm
import math
from math import sqrt
from torchinfo import summary
from torch import Tensor
from typing import Callable, Optional
from torch.utils.checkpoint import checkpoint


class xTransv0_biembed_noreplication(nn.Module):

    def __init__(self, configs):
        super(xTransv0_biembed_noreplication, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.num_nodes = configs.num_nodes
        self.batch_size = configs.batch_size
        
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = int((self.seq_len - self.patch_len)/self.stride + 1)
        self.d_model = configs.d_model
        
        # Embedding
        self.padding_patch = Padding_patch(self.num_nodes, self.patch_num, self.patch_len, self.stride)
        self.input_proj = Input_Projection(self.patch_len, self.d_model)
        self.t_embedding = Temporal_Embedding(self.num_nodes, self.patch_num, self.patch_len, self.d_model, configs.dropout)
        self.s_embedding = Spatial_Embedding(self.num_nodes, self.patch_num, self.patch_len, self.d_model, configs.dropout)

        # Encoder
        self.encoder_t = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), self.d_model, configs.n_heads),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    batchnorm=configs.batch_norm
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model) if configs.norm_layer else None
        )
        
        self.encoder_n = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), self.d_model, configs.n_heads),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    batchnorm=configs.batch_norm
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model) if configs.norm_layer else None
        )
        
        # Decoder
        self.dec = nn.Linear(self.patch_num*self.d_model, self.pred_len)
        self.flatten_dec = Flatten_dec(self.batch_size, self.num_nodes)

    def forward(self, x_batch, x_batch_mark, y_batch_zero, y_batch_mark):
        x_enc, x_mark_enc = x_batch, x_batch_mark
        
        # Normalization from Non-stationary Transformer
        # [B, T, N]
        self.means = torch.mean(x_enc, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc - self.means
        x_enc = x_enc / self.stdev
        x_enc = x_enc.permute(0,2,1)
        
        # Embedding
        x_enc = self.padding_patch(x_enc)  # [B, N, Pnum, Plen]
        x_enc = self.input_proj(x_enc)  # [B, N, Pnum, dmodel]
        enc_in = x_enc
        
        # Encoder
        # t-pos
        enc_t = self.t_embedding(enc_in)
        enc_t = torch.reshape(enc_t, (enc_t.shape[0]*enc_t.shape[1],enc_t.shape[-2],enc_t.shape[-1])) 
        enc_t, _ = self.encoder_t(enc_t)  # [B*N, Pnum, d_model]
        enc_t = torch.reshape(enc_t, (-1, self.num_nodes, enc_t.shape[-2],enc_t.shape[-1]))

        # variable
        # v-pos
        enc_n = self.s_embedding(enc_t).permute(0, 2, 1, 3)
        enc_n = torch.reshape(enc_n, (enc_n.shape[0]*enc_n.shape[1],enc_n.shape[-2],enc_n.shape[-1])) 
        # print("X shape: ", enc_n.shape)
        enc_n, _ = self.encoder_n(enc_n)   # [B*Pnum, N, d_model]
        enc_n = torch.reshape(enc_n, (-1, self.patch_num, enc_n.shape[-2],enc_n.shape[-1]))
        enc_n = enc_n.permute(0, 2, 1, 3)
        enc_out = enc_n
        
        # Flatten and Decoder
        enc_out = self.flatten_dec(enc_out)
        dec_out = self.dec(enc_out).permute(0, 2, 1)
            
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * self.stdev
        dec_out = dec_out + self.means
        
        return dec_out
        


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns



class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu", batchnorm=True):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        if batchnorm:
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # print("X: ", x.shape)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn



class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # print("Attention shape: ", queries.shape, keys.shape, values.shape)  # [B, N, n_heads, d_model/n_heads]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
    

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        _, N, _ = x.shape
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)[:, :N, :]


class Positional_encoding(nn.Module):
    def __init__(self, patch_len, patch_num, d_model, dropout=0.3):
        super(Positional_encoding, self).__init__()
        self.W_P = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        W_pos = torch.empty((patch_num, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True) 
        
    def forward(self, x):
        x = self.W_P(x)
        x = self.dropout(x + self.W_pos)  
        # print("W_pos shape: ", self.W_pos.shape)
        return x


class Padding_patch(nn.Module):
    def __init__(self, nodes, patch_num, patch_len, stride):
        super(Padding_patch, self).__init__()
        self.nodes = nodes
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.stride = stride
        # self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
    def forward(self, x):
        # x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)    # x_enc: [bs x nvars x patch_num x patch_len]
        # x = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[-2],x.shape[-1])) 
        return x


class Flatten_dec(nn.Module):
    def __init__(self, batch_size, nodes):
        super(Flatten_dec, self).__init__()
        self.batch_size = batch_size
        self.nodes = nodes
    def forward(self, x):
        x = torch.reshape(x, (-1, self.nodes, x.shape[-2], x.shape[-1]))  
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[-2]*x.shape[-1]))
        return x


class Input_Projection(nn.Module):
    def __init__(self, patch_len, d_model):
        super(Input_Projection, self).__init__()
        self.linear_proj = nn.Linear(patch_len, d_model)
    def forward(self, x):
        x_linear = self.linear_proj(x)  # [B, N, Pnum, Dmodel]
        return x_linear

class Temporal_Embedding(nn.Module):
    def __init__(self, num_nodes, patch_num, patch_len, d_model, dropout=0.1):
        super(Temporal_Embedding, self).__init__()
        self.patch_embedding = nn.Parameter(
                nn.init.uniform_(torch.empty(patch_num, d_model), -0.02, 0.02),
                requires_grad=True
            )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch, num_nodes, patch_num, _ = x.shape
        x_patch = self.patch_embedding.unsqueeze(0).unsqueeze(1).repeat(batch, num_nodes, 1, 1)  # [B, N, Pnum, Dmodel]
        total = x + x_patch
        total = self.dropout(total)
        return total


class Spatial_Embedding(nn.Module):
    def __init__(self, num_nodes, patch_num, patch_len, d_model, dropout=0.1):
        super(Spatial_Embedding, self).__init__()
        self.node_embedding = nn.Parameter(
                nn.init.uniform_(torch.empty(num_nodes, d_model), -0.02, 0.02),
                requires_grad=True
            )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch, num_nodes, patch_num, _ = x.shape
        x_node = self.node_embedding.unsqueeze(0).unsqueeze(2).repeat(batch, 1, patch_num, 1)   # [B, N, Pnum, Dmodel]
        total = x + x_node
        total = self.dropout(total)
        return total



if __name__ == '__main__':
    
    class Configs(object):
        num_nodes = 7
        num_marks = 4
        seq_len = 96 
        label_len = 48 
        pred_len = 96 
        n_heads = 8
        e_layers = 2 
        d_layers = 1 
        factor = 3 
        enc_in = 7 
        dec_in = 7 
        c_out = 7 
        des = 'Exp' 
        d_model = 16
        d_ff = 128 
        dropout = 0.1
        embed = 'timeF'
        activation = 'gelu'
        output_attention = False
        freq = 'h'
        axis = 'n'
        compress = 1
        patch_len = 1
        stride = 1
        
    configs = Configs()
    model = t_combined(configs)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    # enc = torch.randn([32, configs.seq_len, 7])
    # enc_mark = torch.randn([32, configs.seq_len, 4])
    # dec = torch.randn([32, configs.label_len+configs.pred_len, 7])
    # dec_mark = torch.randn([32, configs.label_len+configs.pred_len, 4])
    
    
    # x = torch.randn([128, configs.seq_len+configs.pred_len, 7, 5])
    # out = model.forward(x)
    
    summary(model, [[32, 96, 7], [32, 96, 4], [32, 96, 7], [32, 96, 4]])