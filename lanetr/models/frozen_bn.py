"""BatchNorm congelado (estilo DETR / Deformable DETR).

En detección, las estadísticas del BatchNorm del backbone se **congelan** (no se actualizan
ni dependen del batch): usa siempre `running_mean`/`running_var` fijos y `weight`/`bias` fijos.
Esto elimina el desajuste train/eval con batches pequeños sin tener que congelar el backbone
entero — sus pesos convolucionales sí pueden entrenar.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d con estadísticas y afines congelados (no aprendibles, no se actualizan)."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        return x * scale + (b - rm * scale)


def freeze_batchnorm(module: nn.Module) -> nn.Module:
    """Sustituye recursivamente todos los `BatchNorm2d` por `FrozenBatchNorm2d` (copiando
    sus estadísticas). Modifica el módulo in-place y lo devuelve."""
    if isinstance(module, nn.BatchNorm2d):
        frozen = FrozenBatchNorm2d(module.num_features, module.eps).to(module.weight.device)
        frozen.weight.data.copy_(module.weight.data)
        frozen.bias.data.copy_(module.bias.data)
        frozen.running_mean.data.copy_(module.running_mean.data)
        frozen.running_var.data.copy_(module.running_var.data)
        return frozen
    for name, child in module.named_children():
        new_child = freeze_batchnorm(child)
        if new_child is not child:
            setattr(module, name, new_child)
    return module
