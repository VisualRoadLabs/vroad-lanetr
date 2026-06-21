"""Codificación de carriles a la representación de FILAS-ANCLA del modelo.

El modelo no predice polilíneas libres, sino, para un conjunto FIJO de filas con `y`
predeterminado, el valor de `x` en cada fila (estilo CLRNet/CLRerNet). Cada *query* del
transformer predice exactamente esto, y la LaneIoU se calcula fila a fila sobre ello.

Por carril codificamos:
    xs      : (R,) x normalizada en [0,1] (x/img_w) en cada fila ancla; válida donde valid=True.
              (puede salir de [0,1] si el carril se sale de pantalla: NO se recorta.)
    valid   : (R,) bool; la fila cae dentro del tramo vertical anotado del carril.
    start   : (x_norm, y_norm) del extremo CERCANO (y máximo, junto al coche).
    length  : extensión vertical del carril / (img_h-1)  -> fracción de filas cubiertas.
    theta   : ángulo (grados) de la dirección cercano->lejano (auxiliar).

`R = num_rows` (144 en el modelo congelado). Las filas van de y=0 (lejos, arriba) a
y=img_h-1 (cerca, abajo). Todo esto trabaja en el **espacio del modelo** (img_w×img_h, 800×320):
la conversión desde la resolución nativa de cada imagen la hace `lanetr.data.format`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ROWS_DEFAULT = 144
IMG_W_DEFAULT = 800
IMG_H_DEFAULT = 320


def make_row_ys(img_h=IMG_H_DEFAULT, num_rows=ROWS_DEFAULT) -> np.ndarray:
    """Posiciones `y` (en píxeles) de las filas ancla: equiespaciadas en [0, img_h-1]."""
    return np.linspace(0.0, img_h - 1, num_rows).astype(np.float32)


@dataclass
class LaneTarget:
    xs: np.ndarray       # (R,) x normalizada
    valid: np.ndarray    # (R,) bool
    start: np.ndarray    # (2,) (x_norm, y_norm) extremo cercano
    length: float
    theta: float         # grados
    slot: int | None


def encode_lane(points: np.ndarray, row_ys: np.ndarray, img_w=IMG_W_DEFAULT,
                img_h=IMG_H_DEFAULT, slot: int | None = None) -> LaneTarget:
    """Polilínea (N,2) en píxeles -> LaneTarget (x por fila ancla)."""
    pts = np.asarray(points, dtype=np.float32)
    ys, xs = pts[:, 1], pts[:, 0]
    order = np.argsort(ys)  # y ascendente para np.interp
    ys_s, xs_s = ys[order], xs[order]

    y_top, y_bottom = float(ys_s[0]), float(ys_s[-1])  # top=lejos, bottom=cerca
    valid = (row_ys >= y_top) & (row_ys <= y_bottom)
    x_at = np.interp(row_ys, ys_s, xs_s)  # fuera de rango -> se fija a los extremos (se enmascara)
    xs_norm = (x_at / img_w).astype(np.float32)

    x_near = float(np.interp(y_bottom, ys_s, xs_s))
    start = np.array([x_near / img_w, y_bottom / (img_h - 1)], np.float32)
    length = (y_bottom - y_top) / (img_h - 1)
    # dirección cercano(bottom) -> lejano(top)
    x_far = float(np.interp(y_top, ys_s, xs_s))
    theta = math.degrees(math.atan2(y_top - y_bottom, x_far - x_near))
    return LaneTarget(xs_norm, valid, start, float(length), float(theta), slot)


def decode_lane(xs_norm: np.ndarray, valid: np.ndarray, row_ys: np.ndarray,
                img_w=IMG_W_DEFAULT) -> np.ndarray:
    """LaneTarget (xs,valid) -> polilínea (M,2) en píxeles, ordenada por y."""
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return np.zeros((0, 2), np.float32)
    x_px = xs_norm[idx] * img_w
    y_px = row_ys[idx]
    return np.stack([x_px, y_px], axis=1).astype(np.float32)


def encode_sample(lanes: list[np.ndarray], slots: list | None, row_ys: np.ndarray,
                  img_w=IMG_W_DEFAULT, img_h=IMG_H_DEFAULT) -> dict:
    """Codifica todos los carriles de una muestra a arrays apilados (L = nº carriles).

    `slots` es opcional: en el **formato común** los carriles no llevan slot (se pasa None y
    todos quedan como -1). Solo CULane nativo aportaba slot 0..3.
    """
    R = len(row_ys)
    slots = slots or []
    targets = [encode_lane(p, row_ys, img_w, img_h, slots[i] if i < len(slots) else None)
               for i, p in enumerate(lanes)]
    L = len(targets)
    return {
        "xs": np.stack([t.xs for t in targets]) if L else np.zeros((0, R), np.float32),
        "valid": np.stack([t.valid for t in targets]) if L else np.zeros((0, R), bool),
        "start": np.stack([t.start for t in targets]) if L else np.zeros((0, 2), np.float32),
        "length": np.array([t.length for t in targets], np.float32),
        "theta": np.array([t.theta for t in targets], np.float32),
        "slots": np.array([(-1 if t.slot is None else t.slot) for t in targets], np.int64),
        "row_ys": row_ys.astype(np.float32),
    }
