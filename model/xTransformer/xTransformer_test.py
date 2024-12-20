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


class t_combined(nn.Module):

    def __init__(self, configs):
        super(t_combined, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.num_nodes = configs.num_nodes
        self.axis = configs.axis
        self.compress = configs.compress
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = int((self.seq_len - self.patch_len)/self.stride + 2)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        
        # Embedding
        self.enc_embedding_pos = positional_encoding(self.patch_len, self.patch_num, configs.d_model, configs.dropout)
        self.enc_embedding_inv = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
       
        
        # Encoder
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Decoder
        self.dec_td = nn.Linear(self.patch_num*configs.d_model, self.pred_len)
        self.dec_d = nn.Linear(configs.d_model, self.pred_len)
        self.dec_final = nn.Linear(self.pred_len, self.pred_len)
        self.dec_dropout = nn.Dropout(0)

    def forward(self, x_batch, x_batch_mark, y_batch_zero, y_batch_mark):
        x_enc, x_mark_enc = x_batch, x_batch_mark
        
        # Normalization from Non-stationary Transformer
        # [B, T, N]
        self.means = torch.mean(x_enc, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc - self.means
        x_enc = x_enc / self.stdev
        
        # Embedding
        if self.axis_t:
            if self.compress:
                print("illegal operation")
            else:
                x_enc_t = x_enc.permute(0,2,1)
                x_enc_t = self.padding_patch_layer(x_enc_t)
                x_enc_t = x_enc.unfold(dimension=-1, size=self.patch_len, step=self.stride)    # x_enc: [bs x nvars x patch_num x patch_len]
                x_enc_t = torch.reshape(x_enc_t, (x_enc_t.shape[0]*x_enc_t.shape[1],x_enc_t.shape[-2],x_enc_t.shape[-1])) 
                x_enc_t = self.enc_embedding_pos(x_enc_t)
                enc_out_t, attns = self.encoder(x_enc_t, x_mark_enc)  # [B, N, d_model]

        if self.axis_n:
            if self.compress:
                x_enc_n = x_enc.permute(0,2,1) 
                x_enc_n = self.enc_embedding_inv(x_enc_n, None)
                # print("X: ", x_enc.shape)
                enc_out_n, attns = self.encoder(x_enc_n, x_mark_enc)  # [B, N, d_model]
            else:
                print("illegal operation")

        # enc_out, attns = self.encoder(x_enc, x_mark_enc)  # [B, N, d_model]
        
        # Flatten
        if self.axis_t:
            if self.compress:
                print("illegal operation")
            else:
                dec_out_t = torch.reshape(enc_out_t, (-1, self.num_nodes, x_enc_t.shape[-2], x_enc_t.shape[-1]))  # [B, N, T, d_model]
                dec_out_t = torch.reshape(dec_out_t, (dec_out_t.shape[0], dec_out_t.shape[1], dec_out_t.shape[-2]*dec_out.shape[-1]))
                dec_out_t = self.dec_td(dec_out_t)

        if self.axis_n:
            if self.compress:
                 dec_out_n = self.dec_d(enc_out_n)
            else:
                print("illegal operation")

        if not self.axis_t:
            dec_out = dec_out_n.permute(0, 2, 1)
        elif not self.axis_n:
            dec_out = dec_out_t.permute(0, 2, 1)
        else:
            dec_out = self.lambda * dec_out_t + (1 - self.lambda) * dec_out_n
            dec_out = self.dec_final(dec_out).permute(0, 2, 1)
        
        # dec_out = self.dec_dropout(dec_out).permute(0, 2, 1)  # [B, T, N]
        # print("dec_out shape: ", dec_out.shape)
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * self.stdev
        dec_out = dec_out + self.means
        # print("Out: ", dec_out.shape)
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
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
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
    
    
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        _, N, _ = x.shape
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)[:, :N, :]


class positional_encoding(nn.Module):
    def __init__(self, patch_len, patch_num, d_model, dropout=0.3):
        super(positional_encoding, self).__init__()
        self.W_P = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        W_pos = torch.empty((patch_num, d_model))
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)  
        nn.init.uniform_(self.W_pos, -0.02, 0.02)
    
    def forward(self, x):
        x = self.W_P(x)
        x = self.dropout(x + self.W_pos)  
        # print("W_pos shape: ", self.W_pos.shape)
        return x

    
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