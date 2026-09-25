"""Full vision-language model.

Wires together the CNN encoder, the adapter (flatten + linear projection
to visual tokens), the Transformer decoder, and the linear head that
produces next-letter logits.
"""

from torch import nn

from configs.config import D_MODEL
from model.decoder import Decoder
from model.encoder import Encoder

class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj = nn.Linear(32, D_MODEL)

    def forward(self, x):
        # x : (B, C, H', W')
        x = nn.flatten(2)       # (B, C, H'*W')
        x = x.transpose(1,2)    # (B, H'*W', C)
        x = self.proj(x)        # (B, H'*W', C) @ (C, D_MODEL) --> (B, H'*W', D_MODEL)

        return x

class VLM(nn.Module):
    def __init__(self):
        super().__init__() 
        self.encoder = Encoder()
        self.adapter = Adapter()
        self.decoder = Decoder(32*32)

    def forward(self, images, input_tokens):
        """
            images = (B, 3, 64, 64)
            input_tokens = (B, T)
        """

        # CNN
        features_map = self.encoder(images)                 # (B, 32, 32, 32)

        # Adapter
        visual_tokens = self.adapter(features_map)          # (B, 32*32, D_MODEL)

        # Decoder
        logits = self.decoder(visual_tokens, input_tokens)  # (B, T, VOCAB_SIZE)

        return logits
