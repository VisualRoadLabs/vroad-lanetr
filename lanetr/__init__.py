"""LaneTR — modelo congelado de detección de carriles para VisualRoad ML Training.

Detector tipo DETR con **atención deformable** sobre un **FPN**, **prior posicional** (anclas)
y **matching húngaro** (1-a-1, sin NMS), entrenado con la **LaneIoU** sensible al ángulo como
coste del matching *y* como pérdida. Salida acotada a ≤ 4 carriles.

Este paquete es la **instancia congelada** del contrato de ML Training: la plataforma lo instala
por SHA (`pip install git+...lanetr@<sha>`) y lo consume **solo a través de `lanetr.contract`**
(`build_model`, `build_criterion`, `MODEL_INFO`, `CONFIG_SCHEMA`, `METRICS_SPEC`, `VISUALIZERS`)
y de `model.predict()`, que emite carriles en el **formato común** `.lines.json`.

La arquitectura interna (backbone DLA-34, FPN, decoder deformable, cabezas, LaneIoU) es **fija**:
aquí no viven experimentos ni ablations; solo el modelo original verificado.
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
