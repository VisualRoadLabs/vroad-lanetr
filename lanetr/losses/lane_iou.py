"""LaneIoU diferenciable, sensible al ángulo — la aportación de la tesis.

Port fiel de CLRerNet (`LaneIoULoss` / `LaneIoUCost`) adaptado a NUESTRA representación de
filas-ancla: los carriles son `xs` (R,) en x normalizada [0,1] con una máscara booleana `valid`
(R,) explícita, en vez del centinela x∉[0,1] de CLRerNet.

Idea (vs. la LineIoU de anchura constante de CLRNet):
  - la anchura virtual de cada fila se **adapta al ángulo local** del carril:
        w_i = lane_width · √(Δx_i² + Δy_i²) / Δy_i  =  lane_width / cos(θ_i)
    En tramos verticales w_i ≈ lane_width; en tramos inclinados se ensancha, igual que la banda
    perpendicular de 30 px de la métrica de evaluación. Por eso LaneIoU correlaciona mejor con F1.

Detalles fieles al original:
  - Δx se mide en píxeles a resolución (img_w=1640, img_h=320) para NO distorsionar el ángulo
    por el resize a 800 (x se comprimió ×0.49). `dy` abarca dos filas (diferencia central).
  - La anchura se calcula con `xs` DETACHED: el gradiente fluye por el centro del carril, no
    por la anchura (estabiliza el entrenamiento).
  - El IoU NO recorta el solape a ≥0 (puede ser negativo si los carriles no solapan), igual
    que CLRNet/CLRerNet — útil como pérdida (rango (−∞,1], en la práctica [−1,1]).
"""
from __future__ import annotations

import torch

# Mitad de anchura virtual por defecto (normalizada por 800), como en CLRerNet.
LANE_WIDTH = 7.5 / 800   # LaneIoU (se ensancha con el ángulo)
IMG_W, IMG_H = 1640, 320
EPS = 1e-9


def _angle_halfwidth(xs: torch.Tensor, lane_width: float, img_w: int, img_h: int) -> torch.Tensor:
    """Mitad de anchura virtual sensible al ángulo, por fila. xs: (..., R) -> (..., R)."""
    R = xs.shape[-1]
    n_strips = R - 1
    dy = img_h / n_strips * 2.0  # span vertical de dos filas (diferencia central)
    dx = (xs[..., 2:] - xs[..., :-2]) * img_w  # Δx central en píxeles
    w = lane_width * torch.sqrt(dx.pow(2) + dy * dy) / dy
    return torch.cat([w[..., :1], w, w[..., -1:]], dim=-1)  # repite bordes -> (..., R)


def _halfwidths(pred_xs, gt_xs, lane_width, img_w, img_h):
    """Anchuras (pred, gt) sensibles al ángulo. La de pred se calcula DETACHED (sin gradiente
    por la anchura)."""
    pw = _angle_halfwidth(pred_xs.detach(), lane_width, img_w, img_h)
    tw = _angle_halfwidth(gt_xs, lane_width, img_w, img_h)
    return pw, tw


def _iou_from_segments(px1, px2, tx1, tx2, valid):
    """IoU 1-D por fila, sumado y enmascarado por `valid`. Devuelve el IoU por par."""
    ovr = torch.minimum(px2, tx2) - torch.maximum(px1, tx1)
    union = torch.maximum(px2, tx2) - torch.minimum(px1, tx1)
    ovr = ovr.masked_fill(~valid, 0.0)
    union = union.masked_fill(~valid, 0.0)
    return ovr.sum(dim=-1) / (union.sum(dim=-1) + EPS)


# --------------------------------------------------------------------------- #
# Versión por pares (matriz P×G) — para el coste del matching húngaro
# --------------------------------------------------------------------------- #
def lane_iou_pairwise(pred_xs, gt_xs, gt_valid, lane_width=LANE_WIDTH, img_w=IMG_W, img_h=IMG_H):
    """Matriz LaneIoU (P×G) entre carriles predichos y GT. Enmascara por filas válidas del GT."""
    pw, tw = _halfwidths(pred_xs, gt_xs, lane_width, img_w, img_h)
    px1, px2 = pred_xs - pw, pred_xs + pw      # (P,R)
    tx1, tx2 = gt_xs - tw, gt_xs + tw          # (G,R)
    # broadcast a (P,G,R)
    px1, px2 = px1[:, None, :], px2[:, None, :]
    tx1, tx2 = tx1[None, :, :], tx2[None, :, :]
    valid = gt_valid[None, :, :].expand(pred_xs.shape[0], -1, -1)
    return _iou_from_segments(px1, px2, tx1, tx2, valid)  # (P,G)


# --------------------------------------------------------------------------- #
# Versión por pares ya emparejados (N) — para la pérdida
# --------------------------------------------------------------------------- #
def _iou_matched(pred_xs, gt_xs, gt_valid, lane_width, img_w, img_h):
    pw, tw = _halfwidths(pred_xs, gt_xs, lane_width, img_w, img_h)
    return _iou_from_segments(pred_xs - pw, pred_xs + pw, gt_xs - tw, gt_xs + tw, gt_valid)


def lane_iou_loss(pred_xs, gt_xs, gt_valid, lane_width=LANE_WIDTH, img_w=IMG_W, img_h=IMG_H,
                  reduction="mean"):
    """Pérdida LaneIoU = 1 − LaneIoU para pares ya emparejados (N,R)."""
    iou = _iou_matched(pred_xs, gt_xs, gt_valid, lane_width, img_w, img_h)
    loss = 1.0 - iou
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def lane_iou_value(pred_xs, gt_xs, gt_valid, lane_width=LANE_WIDTH, img_w=IMG_W, img_h=IMG_H):
    """IoU por par (N,) sin reducir (utilidad para análisis/figuras)."""
    return _iou_matched(pred_xs, gt_xs, gt_valid, lane_width, img_w, img_h)
