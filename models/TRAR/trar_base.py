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
from openvqa.utils.masking import TriangularCausalMask, ProbMask

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

class ProbAttention(nn.Module):
    def __init__(self, __C ,mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    
    
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
#         scores = scores.masked_fill(mask, -1e9)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)
        

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        n_batches = queries.size(0)
#         print(values.shape)
        values = self.linear_v(values).view(
            n_batches,
            -1,
            8,
            #int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            512
        )

        keys = self.linear_k(keys).view(
            n_batches,
            -1,
             8,
            #int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            512
        )
        #k=v

        queries = self.linear_q(queries).view(
            n_batches,
            -1,
             8,
            #int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            512
        )
        
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = 1./ math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.contiguous(), attn
# ---------------------------------
# ---- Multimodal TRAR Decoder ----
# ---------------------------------
class TRAR(nn.Module):
    def __init__(self, __C):
        super(TRAR, self).__init__()

#         self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.conv3 = nn.Conv1d(in_channels=__C.HIDDEN_SIZE, out_channels=__C.HIDDEN_SIZE, kernel_size=1)
        self.attention=ProbAttention(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)
        
        self.avgpool_k = nn.AdaptiveAvgPool2d((1,None))
        self.lin_uk = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.tran_k = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.tran_ck = nn.Linear(__C.HIDDEN_SIZE, 1)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        

    def forward(self, x, y, x_mask, y_mask):
       

       
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=x_mask
        )
        
#         x1=x.contiguous().permute(0, 2, 1)
#         x1 = self.conv3(x1)
#         x1 = x1.transpose(1, 2)
        
        new_x=new_x.view(-1,64,512)
        c_new_x = self.lin_uk(self.avgpool_k(new_x)) # (B, 1, 512)
        new_x = self.linear_k(new_x)  # (B, N, 512)
        merge_new_x = self.tran_k(new_x) + self.tran_ck(c_new_x)
        lamta_new_x = torch.sigmoid(merge_new_x)
        new_x = (1-lamta_new_x) * new_x + lamta_new_x * c_new_x
        
        
        x = self.norm1(x+ self.dropout1(
            new_x
        ))
#         x = self.norm1(x + x1 + self.dropout1(
#             self.mhatt1(v=x, k=x, q=x, mask=x_mask)
#         ))
        x = self.norm2(x + self.dropout2(
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
#         self.endec_list = nn.ModuleList([EnDecoder(__C) for _ in range(__C.LAYER)])

    def forward(self, y, x, y_mask, x_mask):
        

        for enc in self.enc_list:
            y = enc(y, y_mask)


        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)


  
 #         residerX=x
#         residerY=y 
        
#         for endec in self.endec_list:
#             y,x,residerY,residerX=endec(x, y, x_mask, y_mask,residerY,residerX)

        return y, x
    
    def set_tau(self, tau):
        self.tau = tau

    def set_training_status(self, training):
        self.training = training
