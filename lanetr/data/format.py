"""Formato común `.lines.json` + mapeo entre la resolución nativa y el espacio del modelo.

**El formato común manda**: tanto el GT de cualquier dataset como toda predicción
se representan igual, con los puntos en **píxeles de la resolución NATIVA de la imagen**:

```json
{ "timestamp": 1781646771,
  "Lines": [ [ {"x": -11, "y": 550}, {"x": 19, "y": 540}, ... ], ... ],
  "Scores": [0.98, 0.95, ...] }          // opcional; presente solo en predicciones
```

La resolución **no** viaja en el fichero y **no es fija**: usuario = 1280×720 (dashcam),
CULane = 1640×590, otros públicos = su nativa. Se conoce **por la fuente** (constante en usuario;
desde BigQuery `tbl_images.width/height` en públicos). Por eso TODAS las funciones de mapeo reciben
`(src_w, src_h)` explícitos.

El modelo trabaja siempre en **800×320**. Para llegar ahí, igual que en el original de CULane, se
**recorta la franja superior (cielo)** y se **redimensiona** el resto a 800×320. El recorte se
expresa como una **fracción de la altura** (`crop_top_ratio`), de modo que la MISMA receta vale
para cualquier resolución; por defecto = 270/590, que reproduce exactamente el recorte de CULane.

  espacio nativo  --(recorte cielo + resize)-->  espacio modelo 800×320   (encode GT)
  espacio modelo 800×320  --(resize + des-recorte)-->  espacio nativo     (predict)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IMG_W, IMG_H = 800, 320          # espacio del modelo (W×H)
CROP_TOP_RATIO = 270.0 / 590.0   # franja superior recortada (fracción de la altura); = CULane


def crop_top_px(src_h: int, crop_top_ratio: float = CROP_TOP_RATIO) -> int:
    """Píxeles de cielo recortados por arriba para una imagen de altura `src_h`."""
    return int(round(crop_top_ratio * src_h))


def _scales(src_w: int, src_h: int, crop_top_ratio: float, img_w: int, img_h: int):
    """Factores de escala (x, y) y recorte para mapear entre nativo y modelo."""
    crop = crop_top_px(src_h, crop_top_ratio)
    region_h = max(src_h - crop, 1)
    return img_w / src_w, img_h / region_h, crop


def source_to_model(points: np.ndarray, src_w: int, src_h: int,
                    crop_top_ratio: float = CROP_TOP_RATIO,
                    img_w: int = IMG_W, img_h: int = IMG_H) -> np.ndarray:
    """Puntos (N,2) en resolución NATIVA -> espacio del modelo (img_w×img_h).

    Recorta el cielo y redimensiona:  x_m = x·sx ;  y_m = (y - crop)·sy.
    """
    sx, sy, crop = _scales(src_w, src_h, crop_top_ratio, img_w, img_h)
    q = np.asarray(points, dtype=np.float64).copy()
    if len(q) == 0:
        return q.astype(np.float32)
    q[:, 0] = q[:, 0] * sx
    q[:, 1] = (q[:, 1] - crop) * sy
    return q.astype(np.float32)


def model_to_source(points: np.ndarray, src_w: int, src_h: int,
                    crop_top_ratio: float = CROP_TOP_RATIO,
                    img_w: int = IMG_W, img_h: int = IMG_H) -> np.ndarray:
    """Puntos (N,2) en espacio del modelo -> resolución NATIVA (inverso de `source_to_model`).

    x_src = x_m / sx ;  y_src = y_m / sy + crop.  Es el mapeo que usa `predict()` para emitir las
    predicciones en la resolución de origen de cada imagen.
    """
    sx, sy, crop = _scales(src_w, src_h, crop_top_ratio, img_w, img_h)
    q = np.asarray(points, dtype=np.float64).copy()
    if len(q) == 0:
        return q.astype(np.float32)
    q[:, 0] = q[:, 0] / sx
    q[:, 1] = q[:, 1] / sy + crop
    return q.astype(np.float32)


# Lectura / escritura del formato común `.lines.json`
def _as_obj(src) -> dict:
    """Acepta un dict ya parseado, un texto JSON, o una ruta a fichero."""
    if isinstance(src, dict):
        return src
    if isinstance(src, (str, Path)):
        p = Path(src)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return json.loads(str(src))      # texto JSON
    raise TypeError(f"fuente .lines.json no soportada: {type(src)}")


def read_lines_json(src) -> dict:
    """Lee el formato común -> {timestamp, lanes: list[(N,2) float32], scores: list|None}.

    `src` puede ser un dict, un string JSON o una ruta. Los carriles salen como arrays (N,2) en
    píxeles de la resolución NATIVA (tal cual el fichero). `scores` es None si el GT no lo trae.
    """
    obj = _as_obj(src)
    lanes = []
    for lane in obj.get("Lines", []):
        pts = np.array([(float(p["x"]), float(p["y"])) for p in lane], dtype=np.float32)
        lanes.append(pts.reshape(-1, 2))
    scores = obj.get("Scores")
    scores = [float(s) for s in scores] if scores is not None else None
    ts = obj.get("timestamp")
    return {"timestamp": (int(ts) if ts is not None else None), "lanes": lanes, "scores": scores}


def lanes_to_common(lanes: list[np.ndarray], scores: list[float] | None = None,
                    timestamp: int | None = None) -> dict:
    """Construye el dict en formato común a partir de carriles (N,2) en píxeles nativos.

    `x`/`y` se redondean a INTEGER (como el formato). `Scores` se incluye solo si se pasa
    (alineado índice-a-índice con `Lines`); el GT no lo lleva.
    """
    out: dict = {}
    if timestamp is not None:
        out["timestamp"] = int(timestamp)
    out["Lines"] = [
        [{"x": int(round(float(x))), "y": int(round(float(y)))} for x, y in lane]
        for lane in lanes
    ]
    if scores is not None:
        out["Scores"] = [float(s) for s in scores]
    return out


def write_lines_json(path, lanes: list[np.ndarray], scores: list[float] | None = None,
                     timestamp: int | None = None) -> dict:
    """Escribe un `.lines.json` en `path` y devuelve el dict escrito."""
    obj = lanes_to_common(lanes, scores, timestamp)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")
    return obj
