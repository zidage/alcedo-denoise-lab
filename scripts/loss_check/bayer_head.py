"""The authors' UtNet2 with one more stage at the full resolution (runs E4a and E4b).

UtNet2 works at half the resolution (the packed RGGB image) to its last layer, then a 1 x 1
convolution makes 12 channels and PixelShuffle(2) makes the full-resolution RGB image. No layer
sees the mosaic at the full resolution. DemosaicNet has such a stage: after its half-resolution
layers, it puts the mosaic next to the features at the full resolution and applies more
convolutions. Runs E2 and E2b showed that UtNet2 stays far below DemosaicNet on the binned truth
with a 10 times higher learning rate, so run E4 adds such a stage to UtNet2.

The head (FullResolutionHead) gets, at the full resolution:
- the RGB output of UtNet2 (3 channels);
- the last features of UtNet2 (32 channels at half the resolution) after PixelShuffle(2)
  (8 channels);
- the mosaic as three sparse planes (R, G, B; 0 where the color is not measured), 3 channels.

Layers: 3 x 3 convolution to 64 channels, LeakyReLU, 3 x 3 convolution 64 to 64, LeakyReLU, 1 x 1
convolution to 3 channels. The head output is added to the RGB output of UtNet2. The last layer
starts at 0, so at step 0 the model gives the same output as the public weights.

In eval mode, the head runs in tiles of TILE packed px with a margin of MARGIN packed px, so a
full photo fits in the memory of the development machine (10 GB). The receptive field of the head
is 2 full-resolution px, so the tiles give the same output as one pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

HEAD_WIDTH = 64
FEATURE_CHANNELS = 32  # funit of UtNet2
TILE = 512
MARGIN = 4


def sparse_mosaic(packed: torch.Tensor) -> torch.Tensor:
    """(N, 4, h, w) packed RGGB -> (N, 3, 2h, 2w): R, G, B planes, 0 where not measured."""
    zero = torch.zeros_like(packed[:, :1])
    r, g1, g2, b = packed[:, 0:1], packed[:, 1:2], packed[:, 2:3], packed[:, 3:4]
    # PixelShuffle(2): output channel c takes input channels 4c..4c+3 at the positions
    # (0, 0), (0, 1), (1, 0), (1, 1) of each 2 x 2 cell.
    return F.pixel_shuffle(torch.cat([r, zero, zero, zero, zero, g1, g2, zero,
                                      zero, zero, zero, b], dim=1), 2)


class FullResolutionHead(nn.Module):
    def __init__(self, width: int = HEAD_WIDTH):
        super().__init__()
        inputs = 3 + FEATURE_CHANNELS // 4 + 3
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, width, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(width, 3, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, rgb: torch.Tensor, features: torch.Tensor, packed: torch.Tensor):
        x = torch.cat([rgb, F.pixel_shuffle(features, 2), sparse_mosaic(packed)], dim=1)
        return rgb + self.layers(x)


class UtNet2WithHead(nn.Module):
    """UtNet2 (`backbone`) and FullResolutionHead (`head`); same input and output as UtNet2."""

    def __init__(self, backbone: nn.Module, width: int = HEAD_WIDTH):
        super().__init__()
        self.backbone = backbone
        self.head = FullResolutionHead(width)

    def backbone_features(self, packed: torch.Tensor) -> torch.Tensor:
        """UtNet2.forward without the output module (rawnind_jddc raw_denoiser.py)."""
        b = self.backbone
        l1 = b.convs1(b.preprocess(packed))
        l2 = b.convs2(b.maxpool(l1))
        l3 = b.convs3(b.maxpool(l2))
        l4 = b.convs4(b.maxpool(l3))
        x = torch.cat([b.up1(b.bottom(b.maxpool(l4))), l4], dim=1)
        x = torch.cat([b.up2(b.tconvs1(x)), l3], dim=1)
        x = torch.cat([b.up3(b.tconvs2(x)), l2], dim=1)
        x = torch.cat([b.up4(b.tconvs3(x)), l1], dim=1)
        return b.tconvs4(x)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        features = self.backbone_features(packed)
        rgb = self.backbone.output_module(features)
        height, width = packed.shape[-2:]
        if self.training or height * width <= TILE * TILE:
            return self.head(rgb, features, packed)
        output = torch.empty_like(rgb)
        for y0 in range(0, height, TILE):
            for x0 in range(0, width, TILE):
                y1, x1 = min(y0 + TILE, height), min(x0 + TILE, width)
                ya, xa = max(0, y0 - MARGIN), max(0, x0 - MARGIN)
                yb, xb = min(height, y1 + MARGIN), min(width, x1 + MARGIN)
                part = self.head(rgb[..., 2 * ya:2 * yb, 2 * xa:2 * xb],
                                 features[..., ya:yb, xa:xb], packed[..., ya:yb, xa:xb])
                output[..., 2 * y0:2 * y1, 2 * x0:2 * x1] = part[
                    ..., 2 * (y0 - ya):2 * (y1 - ya), 2 * (x0 - xa):2 * (x1 - xa)]
        return output


def model_from_state(raw_denoiser, state: dict) -> nn.Module:
    """UtNet2 or UtNet2WithHead for a state dict of train_bayer.py (the keys tell which)."""
    if any(key.startswith("head.") for key in state):
        model = UtNet2WithHead(raw_denoiser.UtNet2(in_channels=4, funit=FEATURE_CHANNELS,
                                                   activation="LeakyReLU"))
    else:
        model = raw_denoiser.UtNet2(in_channels=4, funit=FEATURE_CHANNELS, activation="LeakyReLU")
    model.load_state_dict(state)
    return model
