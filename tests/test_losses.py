"""Tests de las pérdidas: LaneIoU, matcher húngaro y criterion.

    python tests/test_losses.py        (requiere torch)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from lanetr.data.target_encoding import encode_sample, make_row_ys
from lanetr.losses.criterion import LaneCriterion, prepare_targets
from lanetr.losses.lane_iou import lane_iou_loss, lane_iou_pairwise
from lanetr.losses.matcher import HungarianMatcher

B, NQ, L, R = 2, 12, 6, 144
ROW_YS = make_row_ys(320, R)


def _targets():
    lanes = [np.array([[200, 319], [300, 200], [400, 40]], dtype=np.float32)]
    return [encode_sample(lanes, None, ROW_YS, 800, 320) for _ in range(B)]


def _pred():
    return {
        "conf": torch.randn(L, B, NQ),
        "xs": torch.rand(L, B, NQ, R),
        "start_y": torch.rand(L, B, NQ),
        "length": torch.rand(L, B, NQ),
        "theta": torch.randn(L, B, NQ),
    }


def test_lane_iou_self_is_one():
    xs = torch.rand(3, R)
    valid = torch.ones(3, R, dtype=torch.bool)
    loss = lane_iou_loss(xs, xs, valid, reduction="none")
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-3), loss


def test_lane_iou_pairwise_diagonal_best():
    xs = torch.rand(4, R)
    valid = torch.ones(4, R, dtype=torch.bool)
    iou = lane_iou_pairwise(xs, xs, valid)
    assert torch.argmax(iou, dim=1).tolist() == [0, 1, 2, 3]


def test_matcher_assignment_valid():
    tgt = prepare_targets(_targets(), torch.device("cpu"))
    pred = {k: v[-1] for k, v in _pred().items()}
    m = HungarianMatcher()
    matches = m.match(pred, tgt)
    assert len(matches) == B
    for q, g in matches:
        assert len(q) == len(g) == 1                 # 1 GT por imagen
        assert 0 <= int(q[0]) < NQ


def test_criterion_total_and_backward():
    pred = _pred()
    for v in pred.values():
        v.requires_grad_(True)
    tgt = prepare_targets(_targets(), torch.device("cpu"))
    crit = LaneCriterion()
    losses = crit(pred, tgt)
    assert "total" in losses and losses["total"].ndim == 0
    for k in ("cls", "iou", "xy", "ext"):
        assert k in losses
    losses["total"].backward()
    assert pred["xs"].grad is not None and torch.isfinite(pred["xs"].grad).all()


def test_criterion_handles_empty_gt():
    pred = _pred()
    empty = [{"xs": torch.zeros(0, R), "valid": torch.zeros(0, R, dtype=torch.bool),
              "start_y": torch.zeros(0), "length": torch.zeros(0), "theta": torch.zeros(0)}
             for _ in range(B)]
    losses = LaneCriterion()(pred, empty)
    assert torch.isfinite(losses["total"])


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
