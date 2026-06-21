"""Capa de datos del modelo: codificación a filas-ancla, formato común y transformaciones.

Solo lo que el modelo necesita para definir su **contrato de entrada/salida**:
- `target_encoding` — carriles -> representación de filas-ancla (lo que predice cada query).
- `format` — formato común `.lines.json` + mapeo entre la resolución nativa y el espacio 800×320.
- `transforms` — recorte+resize a 800×320 (cualquier resolución), augment y normalización.

La lectura de datasets concretos (WebDataset, BigQuery, GCS) la hace la plataforma ML Training.
"""
from __future__ import annotations

from . import format, target_encoding, transforms
from .format import (
    CROP_TOP_RATIO,
    IMG_H,
    IMG_W,
    lanes_to_common,
    model_to_source,
    read_lines_json,
    source_to_model,
    write_lines_json,
)
from .target_encoding import decode_lane, encode_lane, encode_sample, make_row_ys
from .transforms import Compose, CropResize, Normalize, build_transforms, denormalize

__all__ = [
    "format", "target_encoding", "transforms",
    "CROP_TOP_RATIO", "IMG_W", "IMG_H",
    "source_to_model", "model_to_source",
    "read_lines_json", "write_lines_json", "lanes_to_common",
    "make_row_ys", "encode_lane", "decode_lane", "encode_sample",
    "build_transforms", "Compose", "CropResize", "Normalize", "denormalize",
]
