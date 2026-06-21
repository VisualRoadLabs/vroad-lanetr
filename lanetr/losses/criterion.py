"""Criterion de LaneTR: la pérdida total de entrenamiento.

Para CADA capa del decoder si `aux_loss` (pérdidas auxiliares estilo DETR), o solo la última:
  1. matching húngaro → empareja queries con carriles GT (1-a-1), con sus costes `cost_*`.
  2. clasificación (focal): queries emparejadas → "carril" (1), el resto → "no-carril" (0).
  3. geometría sobre las parejas: LaneIoU + L1(xs) + L1(start_y, length), con pesos `w_*`.

Devuelve un dict con cada término (para registrar/visualizar) y el `total`. La geometría es
**siempre LaneIoU** (la aportación de la tesis): aquí no hay métricas alternativas ni énfasis
en curvas (son experimentos, no viven en el modelo congelado).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lane_iou import IMG_H, IMG_W, LANE_WIDTH, lane_iou_loss
from .matcher import HungarianMatcher


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
    """Focal loss binaria (sigmoide). logits/targets: misma forma."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t).pow(gamma)
    if alpha >= 0:
        a_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = a_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def prepare_targets(batch_targets, device) -> list[dict]:
    """Convierte los targets del dataset (numpy, de `encode_sample`) a tensores torch en
    `device`, con las claves que usan matcher y criterion."""
    out = []
    for t in batch_targets:
        out.append({
            "xs": torch.as_tensor(t["xs"], dtype=torch.float32, device=device),
            "valid": torch.as_tensor(t["valid"], dtype=torch.bool, device=device),
            "start_y": torch.as_tensor(t["start"][:, 1], dtype=torch.float32, device=device),
            "length": torch.as_tensor(t["length"], dtype=torch.float32, device=device),
            "theta": torch.as_tensor(t["theta"], dtype=torch.float32, device=device),
        })
    return out


class LaneCriterion(nn.Module):
    def __init__(self, matcher: HungarianMatcher | None = None, w_cls=1.0, w_iou=2.0,
                 w_xy=0.5, w_ext=0.5, w_theta=0.0, w_smooth=0.0,
                 cost_cls=1.0, cost_iou=2.0, cost_xy=0.5, cost_ext=0.5,
                 lane_width=LANE_WIDTH, img_w=IMG_W, img_h=IMG_H, aux_loss=True,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.matcher = matcher or HungarianMatcher(cost_cls=cost_cls, cost_iou=cost_iou,
                                                   cost_xy=cost_xy, cost_ext=cost_ext,
                                                   lane_width=lane_width, img_w=img_w, img_h=img_h,
                                                   focal_alpha=focal_alpha, focal_gamma=focal_gamma)
        self.w_cls, self.w_iou, self.w_xy = w_cls, w_iou, w_xy
        self.w_ext, self.w_theta, self.w_smooth = w_ext, w_theta, w_smooth
        self.lane_width, self.img_w, self.img_h = lane_width, img_w, img_h
        self.aux_loss = aux_loss
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma

    def _layer_loss(self, pred_l, targets, matches) -> dict:
        B, NQ = pred_l["conf"].shape
        device = pred_l["conf"].device

        labels = torch.zeros(B, NQ, device=device)
        z = torch.zeros((), device=device)
        iou_l, xy_l, ext_l, th_l, sm_l = z, z, z, z, z
        num = 0
        for b, (q, g) in enumerate(matches):
            if len(q) == 0:
                continue
            labels[b, q] = 1.0
            num += len(q)
            pxs, gxs = pred_l["xs"][b][q], targets[b]["xs"][g]
            gv = targets[b]["valid"][g]
            d = (pxs - gxs).abs().masked_fill(~gv, 0.0)
            l1_pl = d.sum(-1) / gv.sum(-1).clamp(min=1)               # L1 por carril (nb,)
            xy_l = xy_l + l1_pl.sum()
            iou_l = iou_l + lane_iou_loss(pxs, gxs, gv, self.lane_width, self.img_w,
                                          self.img_h, reduction="sum")
            ext_l = ext_l + (pred_l["start_y"][b][q] - targets[b]["start_y"][g]).abs().sum()
            ext_l = ext_l + (pred_l["length"][b][q] - targets[b]["length"][g]).abs().sum()
            th_l = th_l + (pred_l["theta"][b][q] - targets[b]["theta"][g]).abs().sum()
            if self.w_smooth > 0:
                second = pxs[:, 2:] - 2 * pxs[:, 1:-1] + pxs[:, :-2]
                sm_l = sm_l + second.abs().mean(dim=-1).sum()
        n = max(num, 1)

        cls_l = sigmoid_focal_loss(pred_l["conf"], labels, self.focal_alpha,
                                   self.focal_gamma, reduction="sum") / n
        out = {
            "cls": self.w_cls * cls_l,
            "iou": self.w_iou * iou_l / n,
            "xy": self.w_xy * xy_l / n,
            "ext": self.w_ext * ext_l / n,
        }
        if self.w_theta > 0:
            out["theta"] = self.w_theta * th_l / n
        if self.w_smooth > 0:
            out["smooth"] = self.w_smooth * sm_l / n
        return out

    def forward(self, pred, targets) -> dict:
        """`pred`: dict de tensores (L,B,NQ,...). `targets`: lista de B dicts (tensores torch)."""
        L = pred["conf"].shape[0]
        last = L - 1
        layers = range(L) if self.aux_loss else [last]
        totals: dict[str, torch.Tensor] = {}
        for l in layers:
            pred_l = {k: v[l] for k, v in pred.items()}
            matches = self.matcher.match(pred_l, targets)
            for k, v in self._layer_loss(pred_l, targets, matches).items():
                totals[k] = totals.get(k, 0.0) + v
        totals["total"] = sum(totals.values())
        return totals
