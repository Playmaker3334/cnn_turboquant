import torch
import torch.nn as nn
from config import FEATURE_DIM, NUM_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c,  out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN(nn.Module):
    """
    Shared architecture for CNN-A and CNN-B.

    Input  : 3 x 32 x 32
    Block1 : 3   -> 64   | 32x32 -> 16x16
    Block2 : 64  -> 128  | 16x16 ->  8x8
    Block3 : 128 -> 256  |  8x8  ->  4x4
    Pool   : AdaptiveAvgPool -> 256 x 1 x 1
    FC     : 256 -> FEATURE_DIM (256)
    Head   : FEATURE_DIM -> NUM_CLASSES
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(256, FEATURE_DIM)
        self.bn_feat = nn.BatchNorm1d(FEATURE_DIM)
        self.drop    = nn.Dropout(0.3)
        self.head    = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        f = self.bn_feat(self.fc(x))
        if return_features:
            return f
        return self.head(self.drop(f))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
