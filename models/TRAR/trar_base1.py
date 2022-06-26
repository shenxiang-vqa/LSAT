from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

# from local_attention import LocalAttention
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_

# ---------------------------
# ---- Attention Pooling ----
# ---------------------------
class AttFlat(nn.Module):
    def __init__(self, in_channel, glimpses=1, dropout_r=0.1):
        super(AttFlat, self).__init__()
        self.glimpses = glimpses

        self.mlp = MLP(
            in_size=in_channel,
            mid_size=in_channel,
            out_size=glimpses,
            dropout_r=dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            in_channel * glimpses,
            in_channel
        )
        self.norm = LayerNorm(in_channel)

    def forward(self, x, x_mask):
        att = self.mlp(x)

        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        x_atted = self.norm(x_atted)
        return x_atted

# ---------------------------------------
# ---- Multi-Head Attention question ----
# ---------------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            #int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            #int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)
        #k=v

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            #int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)
         
        #attn = LocalAttention(dim=64, window_size=32, causal=False, look_backward=1, look_forward=0, dropout=0.1, exact_windowsize=False)
        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
    
    

# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# -----------------------------
# ---- Transformer Encoder ----
# -----------------------------

class Encoder(nn.Module):
    def __init__(self, __C):
        super(Encoder, self).__init__()

        self.mhattE = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)


    def forward(self, y, y_mask):
        
        y = self.norm1(y + self.dropout1(
            self.mhattE(y, y, y, y_mask)
        ))
       
        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))
        

        return y


# ---------------------------------
# ---- Multimodal TRAR Decoder ----
# ---------------------------------
class TRAR(nn.Module):
    def __init__(self, __C):
        super(TRAR, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)
        


    def forward(self, x, y, x_mask, y_mask):
        

        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))
      
        x = self.norm2(x+ self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))
        
        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))
       
        return x
    



# ----------------------------------------
# ---- Encoder-Decoder with TRAR Block----
# ----------------------------------------
class TRAR_ED(nn.Module):
    def __init__(self, __C):
        super(TRAR_ED, self).__init__()
        self.__C = __C
        
        self.enc_list = nn.ModuleList([Encoder(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([TRAR(__C) for _ in range(__C.LAYER)])


    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        

       
        for dec in self.dec_list:
            x= dec(x, y, x_mask, y_mask)


        return y, x
    
    def set_tau(self, tau):
        self.tau = tau

    def set_training_status(self, training):
        self.training = training
