"""Transformer decoder block and stack.

Masked self-attention + MLP + residual connections + LayerNorm, stacked
N times, with positional embeddings over the combined [visual tokens] +
[letter tokens] sequence.
"""

import torch
import torch.nn as nn

from configs.config import D_MODEL, MAX_TEXT_LEN, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE
from model.attention import Block

class Decoder(nn.Module):
    def __init__(self, num_visual_tokens):
        super().__init__()
        self.text_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.text_position_embedding = nn.Embedding(MAX_TEXT_LEN, D_MODEL)

        self.visual_position_embedding = nn.Embedding(num_visual_tokens, D_MODEL)

        self.blocks = nn.ModuleList([Block(D_MODEL, NUM_HEADS) for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, visual_tokens, input_tokens):
        """
            visual_tokens = (B, S, D_MODEL)
            input_tokens = (B, T)

            result: logits: (B, T, VOCAB_SIZE)
        """

        B,S,_ = visual_tokens.shape
        _,T = input_tokens.shape


        text_tokens = self.text_embedding(input_tokens)
        text_positions = self.text_position_embedding(torch.arange(T))

        visual_positions = self.visual_position_embedding(torch.arange(S))

        text_tokens = text_tokens + text_positions          
        visual_tokens = visual_tokens + visual_positions

        x = torch.cat([visual_tokens, text_tokens], dim=1)  # (B, S+T, D_MODEL)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x[:, S:, :]                 # (B, T, D_MODEL)
        logits = self.lm_head(x)        # (B, T, 27)

        return logits