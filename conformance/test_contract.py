"""Conformance del CONTRATO de ML Training contra el modelo congelado (CLAUDE.md §44).

Verde aquí = "su UI, sus curvas y su viz funcionarán en la plataforma sin tocar nada". Comprueba,
sobre un batch dummy y SIN pesos preentrenados (no baja nada de internet):

  - `build_model(cfg)` -> nn.Module; `forward` saca el dict con las formas esperadas.
  - `model.predict()` saca **formato común** `.lines.json` (Lines + Scores), y mapea a la
    resolución de origen cuando se le pasa `src_sizes` (el requisito de multi-resolución).
  - `build_criterion(cfg)` -> nn.Module; la pérdida da un escalar y el gradiente fluye.
  - `CONFIG_SCHEMA` parsea (path/type/default/group por campo) y `MODEL_INFO` está completo.
  - cada visualizador de `VISUALIZERS` corre y devuelve una imagen.

Ejecutar:  python conformance/test_contract.py     (o: pytest conformance/)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from lanetr.contract.build import build_criterion, build_model, build_param_groups
from lanetr.contract.spec import CONFIG_SCHEMA, METRICS_SPEC, MODEL_INFO
from lanetr.contract.visualizers import VISUALIZERS
from lanetr.data.target_encoding import encode_sample, make_row_ys
from lanetr.losses.criterion import prepare_targets

B, NQ, L, R = 2, 12, 6, 144
CFG = {"arch": {"pretrained": False}}           # sin descargas de pesos


def _dummy_batch():
    x = torch.randn(B, 3, 320, 800)
    row_ys = make_row_ys(320, R)
    pts = np.array([[400, 319], [430, 200], [470, 80]], dtype=np.float32)
    gt = encode_sample([pts], None, row_ys, 800, 320)       # 1 carril GT
    targets = [gt for _ in range(B)]
    return x, targets


def test_model_info_complete():
    for k in ("name", "version", "input", "num_rows", "num_queries", "max_lanes", "common_format"):
        assert k in MODEL_INFO, f"falta {k} en MODEL_INFO"
    assert tuple(MODEL_INFO["input"]) == (3, 320, 800)
    assert MODEL_INFO["common_format"] == ".lines.json"


def test_config_schema_parses():
    groups = {"arch", "optim", "loss", "data", "schedule", "sampler"}
    for f in CONFIG_SCHEMA:
        for k in ("path", "type", "default", "group", "label"):
            assert k in f, f"campo de schema sin {k}: {f}"
        assert f["group"] in groups, f"grupo inválido: {f['group']}"
        assert f["type"] in ("float", "int", "bool", "str", "choice")
        if f["type"] == "choice":
            assert f["default"] in f.get("choices", []), f"default fuera de choices: {f['path']}"
    assert isinstance(METRICS_SPEC, dict) and METRICS_SPEC


def test_build_and_forward_shapes():
    model = build_model(CFG).eval()
    x, _ = _dummy_batch()
    with torch.no_grad():
        pred = model(x)
    assert pred["conf"].shape == (L, B, NQ)
    assert pred["xs"].shape == (L, B, NQ, R)
    for k in ("start_y", "length", "theta"):
        assert pred[k].shape == (L, B, NQ)


def test_predict_common_format():
    model = build_model(CFG)
    x, _ = _dummy_batch()
    out = model.predict(x, conf_thresh=None)                 # None -> todas; se capa a max_lanes
    assert len(out) == B
    for item in out:
        assert "Lines" in item and "Scores" in item
        assert len(item["Lines"]) == len(item["Scores"])
        assert len(item["Lines"]) <= MODEL_INFO["max_lanes"]
        for lane in item["Lines"]:
            for p in lane:
                assert set(p) == {"x", "y"} and isinstance(p["x"], int) and isinstance(p["y"], int)


def test_predict_maps_to_source_resolution():
    """Con src_sizes, los puntos salen en la resolución NATIVA de cada imagen (no en 800×320)."""
    model = build_model(CFG)
    x, _ = _dummy_batch()
    src = [(1640, 590), (1280, 720)]
    out = model.predict(x, src_sizes=src, conf_thresh=None)
    for item, (W, H) in zip(out, src):
        for lane in item["Lines"]:
            for p in lane:
                assert -W <= p["x"] <= 2 * W            # x puede salirse del marco (como CULane)
                assert 0 <= p["y"] <= H                 # y dentro de la altura nativa


def test_criterion_and_grad():
    model = build_model(CFG)
    criterion = build_criterion(CFG)
    x, targets = _dummy_batch()
    pred = model(x)
    tgt = prepare_targets(targets, x.device)
    losses = criterion(pred, tgt)
    assert "total" in losses and losses["total"].ndim == 0
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_param_groups():
    model = build_model(CFG)
    groups = build_param_groups(model, CFG)
    names = {g["name"] for g in groups}
    assert {"rest", "backbone", "slow"} <= names
    lrs = {g["name"]: g["lr"] for g in groups}
    assert lrs["backbone"] < lrs["rest"] and lrs["slow"] < lrs["rest"]


def test_visualizers_return_images():
    model = build_model(CFG).eval()
    x, targets = _dummy_batch()
    with torch.no_grad():
        out = model(x)
    batch = {"images": x, "targets": targets}
    for name, fn in VISUALIZERS.items():
        img = fn(model, batch, out)
        assert isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3, \
            f"visualizador {name} no devolvió imagen RGB"


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} conformance OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
