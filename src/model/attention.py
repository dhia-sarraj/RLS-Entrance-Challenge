"""Handwritten multi-head self-attention.

Q/K/V projections, split heads, scaled dot-product attention, mask,
softmax, merge heads, output projection -- written with tensor
operations, not nn.MultiheadAttention. See tests/test_attention.py for
the required equivalence test against F.scaled_dot_product_attention.

Ressources: Andrej Karpathy - Let's build GPT
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


from configs.config import D_MODEL

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(D_MODEL, head_size, bias=False)
        self.key = nn.Linear(D_MODEL, head_size, bias=False)
        self.value = nn.Linear(D_MODEL, head_size, bias=False)

    def forward(self, x):
        # x : (B, T, D_MODEL) | T = sequence length (46 letters + 1 <EOS>)
        B,T,D_MODEL = x.shape

        k = self.key(x)     # (B,T,D_MODEL) @ (D_MODEL,head_size) --> (B,T,head_size)
        q = self.query(x)   # (B,T,D_MODEL) @ (D_MODEL,head_size) --> (B,T,head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # (B,T,head_size) @ (B,head_size,T) --> (B,T,T)

        # MASK
        mask = torch.tril(
            torch.ones(T, T)
        )
        wei = wei.masked_fill(mask == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)    # (B,T,T)

        v = self.value(x)   # (B,T,D_MODEL) @ (D_MODEL,head_size) --> (B,T,head_size)
        out = wei @ v       # (B,T,T) @ (B,T,head_size) --> (B,T,head_size)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.linear = nn.Linear(n_heads * head_size, D_MODEL)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.linear(out)      # (B,T,n_heads*head_size) @ (n_heads*head_size,D_MODEL) --> (B,T,D_MODEL)

        return out

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL),
            nn.ReLU(),
            nn.Linear(4 * D_MODEL, D_MODEL)
        )

    def forward(self, x):
        # x: (B,T,D_MODEL)
        out = self.nn(x)

        return out

class Block(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        head_size = D_MODEL // n_head
        self.multi_head_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = MLP()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))

        return x
