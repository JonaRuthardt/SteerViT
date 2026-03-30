import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def FeedForward(dim, mult):
    inner_dim = int(dim * mult)
    
    return nn.Sequential(
        nn.LayerNorm(dim), 
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class MaskedCrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            head_dim,
            num_heads,
            ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

        # for extraction
        self.attn_drop = nn.Dropout(0.0)
        self.attn_map = None
        self.save_attn = False  # set True when you want maps

    def forward(self, image_feats, text_feats, attn_mask = None):
        image_feats = self.norm(image_feats)

        if attn_mask is not None:
            attn_mask = attn_mask[:, image_feats.size(1):]      # (B, T_k)
            attn_mask = attn_mask[:, None, None, :]             # (B, 1, 1, T_k) broadcastable

        B, T_q, _ = image_feats.shape
        q = self.to_q(image_feats).reshape(B, T_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        T_k = text_feats.size(1)
        kv = self.to_kv(text_feats).reshape(B, T_k, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        if not self.save_attn:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q * scale) @ k.transpose(-2, -1)  # (B, H, T_q, T_k)

            if attn_mask is not None:
                # attn_mask is boolean where True = keep (as in SDPA boolean masks)
                # convert to additive mask
                neg_inf = torch.finfo(attn.dtype).min
                attn = attn.masked_fill(~attn_mask, neg_inf)

            attn = attn.softmax(dim=-1)
            self.attn_map = attn.detach()
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, T_q, self.inner_dim)
        return self.to_out(out)

class GatedCrossAttention(nn.Module):
    def __init__(
        self,
        layer_idx,
        dim,
        ff_mult,
        use_ffn,
        head_dim=72,
        num_heads=16,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.cross_attn = MaskedCrossAttention(
            dim, 
            head_dim,
            num_heads,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.use_ffn = use_ffn
        if use_ffn:
            self.ff = FeedForward(dim, mult=ff_mult)
            self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, text_feats, attn_mask):
        
        attn_gate = self.attn_gate.tanh()
        x = x + attn_gate * (
            self.cross_attn(
                x,
                text_feats,
                attn_mask = attn_mask
            )
        )
            
        if self.use_ffn:        
            ff_gate = self.ff_gate.tanh()
            x = x + (ff_gate * self.ff(x))
        return x


def attn_forward_wrapper(self, x: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) #3,B,nh,L,D
    q, k, v = qkv.unbind(0) #B,nh,L,D
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        ) # B,nh,L,D
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        self.attn_map = attn #store last layer attn_map during inference
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C) #B,L,nh*D
    x = self.proj(x)
    x = self.proj_drop(x)
    return x