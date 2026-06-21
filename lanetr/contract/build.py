"""Fábricas del contrato: `build_model`, `build_criterion` y utilidades de config/optim.

La plataforma ML Training construye el modelo y la pérdida SOLO a través de aquí, pasando un
`cfg` (dict anidado) = `DEFAULT_CONFIG` fusionado con los overrides `--set` del formulario.
`build_param_groups` expone el conocimiento del modelo sobre qué parámetros van a LR lento
(backbone y deformable+anclas+refinamiento), que el loop de entrenamiento usa al crear el AdamW.
"""
from __future__ import annotations

import copy

import torch.nn as nn

from ..losses.criterion import LaneCriterion
from ..models.frozen_bn import freeze_batchnorm
from ..models.lanetr import LaneTR
from .spec import DEFAULT_CONFIG


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def default_config() -> dict:
    """Copia profunda de la config base efectiva."""
    return copy.deepcopy(DEFAULT_CONFIG)


def merge_config(overrides: dict | None = None) -> dict:
    """`DEFAULT_CONFIG` fusionado con `overrides` (dict anidado)."""
    cfg = default_config()
    return _deep_update(cfg, overrides or {})


def count_parameters(model: nn.Module) -> float:
    """Millones de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def build_model(cfg: dict | None = None) -> nn.Module:
    """Construye el modelo LaneTR desde `cfg['arch']` (capacidad de la familia + campos congelados).

    `arch.ref_refine` ∈ {xs, mlp} → el refinamiento iterativo está SIEMPRE activo, con ese modo.
    Aplica FrozenBatchNorm en el backbone si `cfg['train']['freeze_bn']` (por defecto True).
    """
    cfg = merge_config(cfg)
    a = cfg["arch"]
    model = LaneTR(
        backbone=a["backbone"], pretrained=a["pretrained"], d_model=a["d_model"],
        num_queries=a["num_queries"], num_layers=a["num_decoder_layers"], num_rows=a["num_rows"],
        nhead=a["nhead"], dim_ff=a["dim_ff"], n_points=a["n_points"],
        img_w=a["img_w"], img_h=a["img_h"],
        n_ref_points=a["n_ref_points"], ref_refine=True, ref_refine_mode=a["ref_refine"],
        ref_y_top=a["ref_y_top"], ref_y_bottom=a["ref_y_bottom"],
    )
    model.load_strict = a.get("load_strict", True)   # lo lee el loader de checkpoints del trainer (FT)
    if cfg.get("train", {}).get("freeze_bn", True):
        freeze_batchnorm(model.backbone)
    return model


def build_criterion(cfg: dict | None = None) -> nn.Module:
    """Construye el criterion (focal + LaneIoU + L1) desde `cfg['loss']`.

    Pesos de la pérdida (`w_*`) y costes del matcher (`cost_*`) son independientes; `aux_loss`
    activa las pérdidas auxiliares por capa. La geometría es siempre LaneIoU.
    """
    cfg = merge_config(cfg)
    l = cfg["loss"]
    return LaneCriterion(
        w_cls=l["w_cls"], w_iou=l["w_iou"], w_xy=l["w_xy"], w_ext=l["w_ext"],
        w_theta=l.get("w_theta", 0.0), w_smooth=l.get("w_smooth", 0.0),
        cost_cls=l["cost_cls"], cost_iou=l["cost_iou"], cost_xy=l["cost_xy"],
        cost_ext=l.get("cost_ext", 0.5),
        focal_alpha=l.get("focal_alpha", 0.25), focal_gamma=l["focal_gamma"],
        aux_loss=l.get("aux_loss", True),
    )


def _is_slow(name: str) -> bool:
    """Parámetros que dirigen DÓNDE muestrea la atención (deformable + prior posicional +
    refinamiento de referencias) y que conviene mover despacio (0.1×) para no desestabilizar
    el matching húngaro temprano."""
    return (("sampling_offsets" in name) or ("ref_refine_mlp" in name)
            or name.endswith("anchors.anchors"))


def build_param_groups(model: nn.Module, cfg: dict | None = None) -> list[dict]:
    """Grupos de parámetros con LR diferenciado (backbone ×, deformable+anclas+refine ×, resto 1×).

    Devuelve la lista lista para `torch.optim.AdamW(groups, ...)`. Es conocimiento del modelo
    (qué módulos son 'lentos'); el loop de entrenamiento de ML Training solo lo consume.
    """
    cfg = merge_config(cfg)
    o = cfg["optim"]
    lr, bb, slow = o["lr"], o["backbone_lr_mult"], o.get("slow_mult", 0.1)
    g_bb, g_slow, g_rest = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone"):
            g_bb.append(p)
        elif _is_slow(name):
            g_slow.append(p)
        else:
            g_rest.append(p)
    groups = [
        {"params": g_rest, "lr": lr, "name": "rest"},
        {"params": g_bb, "lr": lr * bb, "name": "backbone"},
        {"params": g_slow, "lr": lr * slow, "name": "slow"},
    ]
    return [g for g in groups if len(g["params"]) > 0]
