
__all__ = ['PatchTST']


from typing import Optional
import torch
import math
import numpy as np
from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchTST(nn.Module):
    '''
    A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR 2023.
    https://arxiv.org/pdf/2211.14730
    https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_self_supervised/src/models/patchTST.py
    '''
    def __init__(self,
                 num_channels: int,
                 patch_size: int,
                 num_patch: int,
                 n_layers: int = 3,
                 d_model: int = 128,
                 n_heads: int = 16,
                 shared_embedding: bool = True,
                 d_ff: int = 256,
                 norm:str='BatchNorm',
                 attn_dropout: float = 0.,
                 dropout: float = 0.,
                 act: str = "gelu",
                 res_attention: bool = True,
                 pre_norm: bool = False,
                 store_attn: bool = False,
                 pe: str = 'zeros',
                 learn_pe: bool = True,
                 head_dropout: float = 0,
                 num_classes: int = None,
                 **kwargs):

        super().__init__()

        self.patchify = Rearrange('b c (n p) -> b n c p', p = patch_size)
        self.unpatchify = Rearrange('b n c p -> b c (n p)')

        # Backbone
        self.backbone = PatchTSTEncoder(
            num_channels, num_patch=num_patch, patch_size=patch_size,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            shared_embedding=shared_embedding, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout, act=act,
            res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
            pe=pe, learn_pe=learn_pe,
            **kwargs)

        # Head
        self.predictor = PredictionHead(
            d_model=d_model, patch_size=patch_size, head_dropout=head_dropout)
        self.classifier = ClassificationHead(
            num_channels=num_channels, d_model=d_model, num_classes=num_classes, head_dropout=head_dropout)

    def encode(self, x):
        '''
        x: tensor [bs, num_channels, seq_len], where seq_len = num_patch * patch_size
        '''
        x = self.patchify(x)                    # [bs, num_patch, num_channels, patch_size] (b n c p)
        z = self.backbone(x)                    # [bs, num_channels, d_model, num_patch]    (b c d n)
        z = rearrange(z, 'b c d n -> b n c d')  # [bs, num_patch, num_channels, d_model]    (b n c d)
        return z

    def predict(self, z):
        '''
        tensor [bs, num_patch, num_channels, d_model]
        '''
        z = self.predictor(z)                   # [bs, num_patch, num_channels, patch_size] (b n c p)
        x = self.unpatchify(z)                  # [bs, num_patch, seq_len]                  (b c (n p))
        return x

    def classify(self, z):
        y = self.classifier(z)                  # [bs, num_channels, num_classes]           (b c k)
        return y

    def forward(self, x):
        z = self.encode(x)
        x = self.predict(z)
        y = self.classify(z)
        return x, y


class PredictionHead(nn.Module):
    def __init__(self, d_model, patch_size, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_size)

    def forward(self, z):
        '''
        z: [bs, num_patch, num_channels, d_model]
        output: [bs, num_classes]
        '''
        z = self.dropout(z)                    # [bs, num_patch, num_channels, d_model]
        z = self.linear(z)                     # [bs, num_patch, num_channels, patch_size]
        return z

class ClassificationHead(nn.Module):
    def __init__(self, num_channels, d_model, num_classes, head_dropout: float = 0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(num_channels * d_model, num_classes)

    def forward(self, z):
        '''
        z: [bs, num_patch, num_channels, d_model]
        output: [bs, num_classes]
        '''
        z = rearrange(z, 'b n c d -> b c d n')  # [bs, num_channels, d_model, num_patch]
        z = z[:, :, :, -1]                      # [bs, num_channels, d_model]
        z = self.flatten(z)                     # [bs, num_channels * d_model]
        z = self.dropout(z)                     # [bs, num_channels * d_model]
        y = self.linear(z)                      # [bs, num_classes]
        return y


class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_size,
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, **kwargs):

        super().__init__()
        self.num_channels = c_in
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.d_model = d_model
        self.shared_embedding = shared_embedding

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.num_channels): self.W_P.append(nn.Linear(patch_size, d_model))
        else:
            self.W_P = nn.Linear(patch_size, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)

    def forward(self, x) -> Tensor:
        '''
        x: tensor [bs, num_patch, num_channels, patch_size]
        '''
        bs, num_patch, num_channels, _ = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(num_channels):
                z = self.W_P[i](x[: ,: ,i, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)                                                      # x: [bs, num_patch, num_channels, d_model]
        x = x.transpose(1, 2)                                                    # x: [bs, num_channels, num_patch, d_model]

        u = torch.reshape(x, (bs * num_channels, num_patch, self.d_model))       # u: [bs * num_channels, num_patch, d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * num_channels, num_patch, d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * num_channels, num_patch, d_model]
        z = torch.reshape(z, (-1, num_channels, num_patch, self.d_model))        # z: [bs, num_channels, num_patch, d_model]
        z = z.permute(0, 1, 3, 2)                                                # z: [bs, num_channels, d_model, num_patch]

        return z


class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                attn_dropout=attn_dropout, dropout=dropout,
                activation=activation, res_attention=res_attention,
                pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        '''
        src: tensor [bs x q_len x d_model]
        '''
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output

class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None):
        '''
        src: tensor [bs x q_len x d_model]
        '''
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False,
                 attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        '''Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        '''
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class ScaledDotProductAttention(nn.Module):
    r'''
    Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017)
    with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and
    locality self sttention (Vision Transformer for Small-Size Datasets by Lee et al, 2021)
    '''

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                        # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                                 # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                           # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high
    def forward(self, x):
        return torch.sigmoid(x) * (self.high - self.low) + self.low

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


if __name__ == '__main__':

    seq_len = 500
    patch_size = 10
    num_patch = seq_len // patch_size

    model = PatchTST(
        num_channels=3,
        patch_size=patch_size,
        num_patch=num_patch,
        n_layers=2,
        n_heads=16,
        d_model=128,
        shared_embedding=True,
        d_ff=512,
        dropout=0.2,
        head_dropout=0.2,
        act='relu',
        res_attention=False,
        num_classes=1000)

    time_series = torch.randn(4, 3, seq_len)
    x, y = model(time_series)

    print(f'reconstruction shape: {x.shape}')
    print(f'classification shape: {y.shape}')
