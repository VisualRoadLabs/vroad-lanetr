"""Tests del formato común y del mapeo de resolución (numpy puro, sin torch).

    python tests/test_format.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lanetr.data.format import (
    CROP_TOP_RATIO,
    lanes_to_common,
    model_to_source,
    read_lines_json,
    source_to_model,
)


def test_resolution_roundtrip_culane():
    """nativo -> modelo -> nativo recupera los puntos (CULane 1640×590)."""
    pts = np.array([[10, 580], [800, 400], [1639, 300]], dtype=np.float32)
    m = source_to_model(pts, 1640, 590)
    back = model_to_source(m, 1640, 590)
    assert np.allclose(back, pts, atol=1e-3), back


def test_resolution_roundtrip_user():
    """Lo mismo para la resolución de usuario (1280×720): redimensiona a la común y vuelve."""
    pts = np.array([[0, 700], [640, 500], [1279, 360]], dtype=np.float32)
    m = source_to_model(pts, 1280, 720)
    back = model_to_source(m, 1280, 720)
    assert np.allclose(back, pts, atol=1e-3), back


def test_model_space_is_800x320():
    """Un punto del borde inferior-derecho nativo cae dentro del marco del modelo 800×320."""
    pts = np.array([[1639, 589]], dtype=np.float32)
    m = source_to_model(pts, 1640, 590)
    assert 0 <= m[0, 0] <= 800 + 1
    assert 0 <= m[0, 1] <= 320 + 1


def test_crop_removes_sky():
    """La franja de cielo (por encima del recorte) mapea a y negativa en el espacio del modelo."""
    crop_px = CROP_TOP_RATIO * 590
    sky = np.array([[800, crop_px - 50]], dtype=np.float32)   # por encima del recorte
    m = source_to_model(sky, 1640, 590)
    assert m[0, 1] < 0, m


def test_common_format_roundtrip():
    """lanes -> dict común -> lanes recupera la geometría (con redondeo a int)."""
    lanes = [np.array([[-11, 550], [19, 540], [769, 290]], dtype=np.float32)]
    obj = lanes_to_common(lanes, scores=[0.97], timestamp=1781646771)
    assert obj["timestamp"] == 1781646771
    assert obj["Scores"] == [0.97]
    parsed = read_lines_json(obj)
    assert parsed["timestamp"] == 1781646771
    assert np.allclose(parsed["lanes"][0], lanes[0], atol=1.0)


def test_common_format_gt_has_no_scores():
    """El GT no lleva Scores; las predicciones sí."""
    lanes = [np.array([[100, 300], [200, 100]], dtype=np.float32)]
    gt = lanes_to_common(lanes)                      # sin scores
    assert "Scores" not in gt
    pred = lanes_to_common(lanes, scores=[0.8])
    assert pred["Scores"] == [0.8]


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
    print(f"\n{len(tests) - failed}/{len(tests)} tests OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
