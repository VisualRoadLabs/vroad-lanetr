"""FPN (Feature Pyramid Network): fusiona los 3 niveles del backbone en una pirámide
de features con un número común de canales (256 por defecto).

Camino top-down clásico: convolución lateral 1×1 en cada nivel para igualar canales,
suma del nivel superior (upsampleado) y convolución 3×3 de salida. Devuelve [P3, P4, P5]
a strides 8/16/32, todos con `out_channels`. Estos mapas son la "memoria" que mirará el
decoder transformer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels_list: list[int], out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1)
                                      for c in in_channels_list])
        self.output = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                                     for _ in in_channels_list])
        self.out_channels = out_channels
        self.num_levels = len(in_channels_list)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(feats) == self.num_levels, f"esperaba {self.num_levels} niveles, recibí {len(feats)}"
        laterals = [conv(f) for conv, f in zip(self.lateral, feats)]
        # top-down: del nivel más profundo (C5) al más superficial (C3)
        for i in range(self.num_levels - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up
        return [conv(l) for conv, l in zip(self.output, laterals)]
