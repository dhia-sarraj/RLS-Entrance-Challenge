"""CNN vision encoder.

Built from nn.Conv2d, activations, normalization and pooling (no
pretrained backbones). Turns a (B, 3, 64, 64) image into a feature map
(B, C, H', W').
"""

from torch import nn
from configs.config import KERNEL_SIZE, NUM_FILTERS

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=NUM_FILTERS, kernel_size=KERNEL_SIZE, padding=1), # (3, 64, 64) --> (32, 64, 64) | FORMULA: ouput = (input + 2*padding - kernel_size) / stride + 1 ==> (64 + 2*1 - 3) / 1 + 1 = 64 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (32, 64, 64) --> (32, 32, 32)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x