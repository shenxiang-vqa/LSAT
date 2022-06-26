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
    
    
# ---------------------------------------
# ---- ME----
# ---------------------------------------
    
class MHAttE(nn.Module):
    def __init__(self, __C):
        super(MHAttE, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        
#         self.avgpool_so=nn.AdaptiveAvgPool2d((1,None))
#         self.avgpool_pr=nn.AdaptiveAvgPool2d((1,None))
#         self.lin_so=nn.Linear(14,14)
#         self.lin_pr=nn.Linear(14,14)
#         self.tran_so=nn.Linear(14,1)
#         self.tran_pr=nn.Linear(14,1)
#         self.tran_cso=nn.Linear(14,1)
#         self.tran_cpr=nn.Linear(14,1)
        
#         self.lin_so=nn.Linear(100,100)
#         self.lin_pr=nn.Linear(100,100)
#         self.tran_so=nn.Linear(100,1)
#         self.tran_pr=nn.Linear(100,1)
#         self.tran_cso=nn.Linear(100,1)
#         self.tran_cpr=nn.Linear(100,1)
        
#         self.resweight = nn.Parameter(torch.Tensor([0.2]))

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
        
        
        global prev         
        if prev is not None:
            scores=scores+prev
        prev=scores

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

class EncoderY(nn.Module):
    def __init__(self, __C):
        super(EncoderY, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        
        y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y

# -----------------------------
# ---- Transformer EncoderX ----
# -----------------------------

class EncoderX(nn.Module):
    def __init__(self, __C):
        super(EncoderX, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x



    
class EnDecoder(nn.Module):
    def __init__(self, __C):
        super(EnDecoder, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.mhatt3 = MHAtt(__C)
        self.mhatt4 = MHAtt(__C)
        self.ffn1 = FFN(__C)
        self.ffn2 = FFN(__C)
        self.ffn3 = FFN(__C)
        self.ffn4 = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout4 = nn.Dropout(__C.DROPOUT_R)
        self.norm4 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout5 = nn.Dropout(__C.DROPOUT_R)
        self.norm5 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout6 = nn.Dropout(__C.DROPOUT_R)
        self.norm6 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout7 = nn.Dropout(__C.DROPOUT_R)
        self.norm7 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout8 = nn.Dropout(__C.DROPOUT_R)
        self.norm8 = LayerNorm(__C.HIDDEN_SIZE)

        
    def forward(self, x, y, x_mask, y_mask,residualY,residualX):
    
 
        # SA-en

        y = self.norm1(y+residualY+ self.dropout1(
           self.mhatt1(y, y, y, y_mask) 
        ))

        y = self.norm2(y+ self.dropout2(
          self.ffn1(y)
        ))
        residualY=y
        # SA-de

        x = self.norm3(x +residualX+self.dropout3(
            self.mhatt2(v=x, k=x, q=x, mask=x_mask)
        ))

        x = self.norm4(x + self.dropout4(
            self.ffn2(x)
        ))
        residualX=x
        
                
            
        # GA-en

        y = self.norm5(y +self.dropout5(
            self.mhatt3(v=x, k=x, q=y, mask=x_mask)
        ))
        y = self.norm6(y + self.dropout6(
            self.ffn3(y)
        ))

        
  
        # GA-de
        x = self.norm7(x+self.dropout7(
            self.mhatt4(v=y, k=y, q=x, mask=y_mask)
        ))

        
        x = self.norm8(x + self.dropout8(
            self.ffn4(x)
        ))
        

        return y,x,residualY,residualX
    


# ----------------------------------------
# ---- Encoder-Decoder with TRAR Block----
# ----------------------------------------
class TRAR_ED(nn.Module):
    def __init__(self, __C):
        super(TRAR_ED, self).__init__()
        self.__C = __C
        
#         self.encY_list = nn.ModuleList([EncoderY(__C) for _ in range(__C.LAYER)])
#         self.encX_list = nn.ModuleList([EncoderX(__C) for _ in range(__C.LAYER)])
#         self.dec_list = nn.ModuleList([TRAR(__C) for _ in range(__C.LAYER)])
        self.endec_list = nn.ModuleList([EnDecoder(__C) for _ in range(__C.LAYER)])

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
 
        
#         for encY in self.encY_list:
#             y = encY(y, y_mask)
            
#         for encX in self.encX_list:
#             x = encX(x, x_mask)

#         # Input encoder last hidden vector
#         # And obtain decoder last hidden vectors
#         for dec in self.dec_list:
#             x = dec(x, y, x_mask, y_mask)
#         return y, x



        residualY=y
        residualX=x
        
        for endec in self.endec_list:
            y,x,residualY,residualX=endec(x, y, x_mask, y_mask,residualY,residualX)

        return y, x
    
    def set_tau(self, tau):
        self.tau = tau

    def set_training_status(self, training):
        self.training = training
