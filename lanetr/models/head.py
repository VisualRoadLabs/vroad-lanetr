"""Cabezas de predicción por query.

Convierten cada vector de query (d_model) que sale del decoder en una predicción de carril,
en la representación de filas-ancla:

    conf     : 1 logit    -> ¿esta query contiene un carril? (confianza)
    xs       : R valores   -> x normalizada [0,1] en cada una de las R filas-ancla
    start_y  : 1 valor [0,1] -> y del extremo CERCANO (abajo) del carril
    length   : 1 valor [0,1] -> extensión vertical del carril (fracción de la imagen)
    theta    : 1 valor     -> ángulo (auxiliar, ligado al prior posicional)

Las cabezas se aplican a la salida de TODAS las capas del decoder (para pérdidas auxiliares);
en inferencia se usa la última capa.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data.target_encoding import make_row_ys


class MLP(nn.Module):
    """Perceptrón multicapa simple (estilo DETR)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class LaneHead(nn.Module):
    def __init__(self, d_model: int = 256, num_rows: int = 144, residual_xs: bool = False):
        super().__init__()
        self.num_rows = num_rows
        self.residual_xs = residual_xs
        self.conf = nn.Linear(d_model, 1)
        self.xs = MLP(d_model, d_model, num_rows)
        self.ext = MLP(d_model, d_model, 2)   # (start_y, length)
        self.theta = nn.Linear(d_model, 1)
        if residual_xs:
            # delta≈0 al inicio -> xs y extensión ≈ las anclas hasta que el modelo aprenda
            for head in (self.xs, self.ext):
                nn.init.zeros_(head.layers[-1].weight)
                nn.init.zeros_(head.layers[-1].bias)

    def forward(self, hs: torch.Tensor, prior_xs: torch.Tensor | None = None,
                prior_ext: torch.Tensor | None = None) -> dict:
        """hs: (L, B, NQ, D) -> dict de tensores (L, B, NQ, ...).

        Con anclas (`residual_xs`): xs = prior_xs + delta (puede salir de [0,1] para carriles
        fuera de pantalla); extensión = sigmoid(logit(prior_ext) + delta) (en [0,1], parte del
        prior). Sin anclas: xs = sigmoid(delta), extensión = sigmoid(delta).
        """
        delta_xs = self.xs(hs)
        if self.residual_xs and prior_xs is not None:
            xs = prior_xs.view(1, 1, *prior_xs.shape) + delta_xs   # (L,B,NQ,R)
        else:
            xs = delta_xs.sigmoid()

        ext_delta = self.ext(hs)
        if self.residual_xs and prior_ext is not None:
            base = _inverse_sigmoid(prior_ext).view(1, 1, *prior_ext.shape)  # (1,1,NQ,2)
            ext = (base + ext_delta).sigmoid()
        else:
            ext = ext_delta.sigmoid()

        return {
            "conf": self.conf(hs).squeeze(-1),        # (L,B,NQ) logits
            "xs": xs,                                 # (L,B,NQ,R)
            "start_y": ext[..., 0],                   # (L,B,NQ) en [0,1]
            "length": ext[..., 1],                    # (L,B,NQ) en [0,1]
            "theta": self.theta(hs).squeeze(-1),      # (L,B,NQ)
        }


@torch.no_grad()
def decode_lanes(pred: dict, layer: int = -1, conf_thresh: float | None = 0.5,
                 num_rows: int = 144, img_w: int = 800, img_h: int = 320) -> list[list[dict]]:
    """Convierte la salida del modelo en carriles dibujables (espacio img_w×img_h).

    Devuelve, por imagen del batch, una lista de dicts {points (M,2), conf, query}.
    Si `conf_thresh` es None, devuelve todas las queries (útil para visualizar candidatos).
    """
    row_ys = make_row_ys(img_h, num_rows)
    conf = pred["conf"][layer].sigmoid().cpu().numpy()   # (B,NQ)
    xs = pred["xs"][layer].cpu().numpy()                 # (B,NQ,R)
    start_y = pred["start_y"][layer].cpu().numpy()       # (B,NQ)
    length = pred["length"][layer].cpu().numpy()         # (B,NQ)
    B, NQ = conf.shape

    out = []
    for b in range(B):
        lanes = []
        for q in range(NQ):
            if conf_thresh is not None and conf[b, q] < conf_thresh:
                continue
            y_bottom = start_y[b, q] * (img_h - 1)
            y_top = y_bottom - length[b, q] * (img_h - 1)
            lo, hi = min(y_top, y_bottom), max(y_top, y_bottom)
            idx = np.where((row_ys >= lo) & (row_ys <= hi))[0]
            if len(idx) < 2:
                continue
            pts = np.stack([xs[b, q, idx] * img_w, row_ys[idx]], axis=1).astype(np.float32)
            lanes.append({"points": pts, "conf": float(conf[b, q]), "query": q})
        out.append(lanes)
    return out
