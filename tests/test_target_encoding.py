"""Tests de la codificación a filas-ancla (numpy puro, sin torch).

    python tests/test_target_encoding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from lanetr.data.target_encoding import (
    decode_lane,
    encode_lane,
    encode_sample,
    make_row_ys,
)

R, IMG_W, IMG_H = 144, 800, 320
ROW_YS = make_row_ys(IMG_H, R)


def test_row_ys_span():
    assert ROW_YS[0] == 0.0 and abs(ROW_YS[-1] - (IMG_H - 1)) < 1e-4
    assert len(ROW_YS) == R


def test_encode_decode_roundtrip():
    """Codificar y decodificar una recta recupera la polilínea (en las filas cubiertas)."""
    pts = np.array([[200, 319], [300, 200], [400, 40]], dtype=np.float32)
    t = encode_lane(pts, ROW_YS, IMG_W, IMG_H)
    dec = decode_lane(t.xs, t.valid, ROW_YS, IMG_W)
    # cada y decodificado coincide con la x interpolada del original
    ys, xs = pts[:, 1], pts[:, 0]
    order = np.argsort(ys)
    for x, y in dec:
        x_ref = np.interp(y, ys[order], xs[order])
        assert abs(x - x_ref) < 2.0, (x, x_ref, y)


def test_valid_mask_within_extent():
    """`valid` solo marca las filas dentro del tramo vertical anotado."""
    pts = np.array([[200, 280], [260, 160]], dtype=np.float32)  # de y=160 a y=280
    t = encode_lane(pts, ROW_YS, IMG_W, IMG_H)
    covered = ROW_YS[t.valid]
    assert covered.min() >= 160 - 1 and covered.max() <= 280 + 1


def test_encode_sample_shapes_and_no_slots():
    """encode_sample sin slots (formato común): todos los slots quedan a -1."""
    lanes = [np.array([[100, 319], [150, 100]], dtype=np.float32),
             np.array([[600, 319], [550, 100]], dtype=np.float32)]
    s = encode_sample(lanes, None, ROW_YS, IMG_W, IMG_H)
    assert s["xs"].shape == (2, R)
    assert s["valid"].shape == (2, R)
    assert s["start"].shape == (2, 2)
    assert list(s["slots"]) == [-1, -1]


def test_encode_empty_sample():
    s = encode_sample([], None, ROW_YS, IMG_W, IMG_H)
    assert s["xs"].shape == (0, R)


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
