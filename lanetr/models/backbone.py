"""Backbone CNN: extrae features multiescala (C3, C4, C5) de la imagen.

Por defecto **DLA-34** preentrenado en ImageNet, que es el backbone de
CLRNet. Devuelve los mapas a strides 8, 16, 32 con canales [128, 256, 512], que
alimentan el FPN. También se puede usar ResNet-18/34/50 (vía torchvision) con la misma
interfaz: `build_backbone('resnet18', ...)`.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

# canales de salida [C3, C4, C5] de las ResNet de torchvision
_RESNET_CHANNELS = {
    "resnet18": [128, 256, 512],
    "resnet34": [128, 256, 512],
    "resnet50": [512, 1024, 2048],
}


class ResNetBackbone(nn.Module):
    """ResNet de torchvision que devuelve [C3, C4, C5] (strides 8/16/32)."""

    def __init__(self, name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if name not in _RESNET_CHANNELS:
            raise ValueError(f"backbone resnet desconocido: {name}")
        from torchvision import models

        weights_enum = {
            "resnet18": models.ResNet18_Weights,
            "resnet34": models.ResNet34_Weights,
            "resnet50": models.ResNet50_Weights,
        }[name]
        weights = weights_enum.IMAGENET1K_V1 if pretrained else None
        try:
            net = getattr(models, name)(weights=weights)
        except Exception as e:  # sin internet para descargar pesos
            if pretrained:
                warnings.warn(f"No se pudieron cargar pesos preentrenados ({e}); pesos aleatorios.")
                net = getattr(models, name)(weights=None)
            else:
                raise

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)  # stride 4
        self.layer1 = net.layer1  # stride 4
        self.layer2 = net.layer2  # stride 8  -> C3
        self.layer3 = net.layer3  # stride 16 -> C4
        self.layer4 = net.layer4  # stride 32 -> C5
        self.out_channels = list(_RESNET_CHANNELS[name])
        self.strides = [8, 16, 32]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]


class TimmBackbone(nn.Module):
    """Backbone de `timm` en modo `features_only` (p.ej. DLA-34).

    Construye la red completa y selecciona los niveles con stride 8/16/32 por su reducción
    real (no por `out_indices`, cuya semántica varía entre modelos). Para DLA-34 eso da
    canales [128, 256, 512], igual que la ResNet -> el FPN no cambia.
    """

    def __init__(self, name: str = "dla34", pretrained: bool = True, strides=(8, 16, 32)):
        super().__init__()
        import timm

        try:
            self.model = timm.create_model(name, features_only=True, pretrained=pretrained)
        except Exception as e:  # sin internet para descargar pesos
            if pretrained:
                warnings.warn(f"No se pudieron cargar pesos preentrenados ({e}); pesos aleatorios.")
                self.model = timm.create_model(name, features_only=True, pretrained=False)
            else:
                raise

        reductions = list(self.model.feature_info.reduction())
        channels = list(self.model.feature_info.channels())
        self._sel = [i for i, r in enumerate(reductions) if r in strides]
        if len(self._sel) != len(strides):
            raise RuntimeError(f"{name}: strides {strides} no encontrados en {reductions}")
        self.out_channels = [channels[i] for i in self._sel]
        self.strides = [reductions[i] for i in self._sel]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.model(x)
        return [feats[i] for i in self._sel]


def build_backbone(name: str = "dla34", pretrained: bool = True) -> nn.Module:
    """Crea el backbone. ResNet* -> torchvision; cualquier otro (p.ej. dla34) -> timm."""
    if name in _RESNET_CHANNELS:
        return ResNetBackbone(name, pretrained)
    return TimmBackbone(name, pretrained)
