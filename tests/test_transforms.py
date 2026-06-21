"""Tests de las transformaciones (recorte+resize a la resolución común, normalización, targets).

    python tests/test_transforms.py        (requiere torch + Pillow)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from lanetr.data.format import source_to_model
from lanetr.data.transforms import CropResize, build_transforms


def _sample(W, H):
    img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
    lanes = [np.array([[W // 4, H - 5], [W // 2, H // 2], [3 * W // 4, int(H * 0.5)]], np.float32)]
    return {"image": img, "lanes": lanes, "meta": {}}


def test_cropresize_to_common_size():
    """Cualquier resolución de entrada acaba en 800×320."""
    for (W, H) in [(1640, 590), (1280, 720), (1920, 1080)]:
        s = CropResize(800, 320)(_sample(W, H), np.random.default_rng(0))
        assert s["image"].size == (800, 320)
        assert s["meta"]["src_size"] == (W, H)


def test_cropresize_matches_format_mapping():
    """CropResize transforma los carriles igual que `format.source_to_model`."""
    W, H = 1640, 590
    s0 = _sample(W, H)
    pts0 = s0["lanes"][0].copy()
    s = CropResize(800, 320)(s0, np.random.default_rng(0))
    expected = source_to_model(pts0, W, H)
    assert np.allclose(s["lanes"][0], expected, atol=1e-3)


def test_build_transforms_produces_tensor_and_targets():
    tf = build_transforms("val", encode_targets=True)
    s = tf(_sample(1280, 720), np.random.default_rng(0))
    assert torch.is_tensor(s["image"]) and s["image"].shape == (3, 320, 800)
    assert "targets" in s and s["targets"]["xs"].shape[1] == 144


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
