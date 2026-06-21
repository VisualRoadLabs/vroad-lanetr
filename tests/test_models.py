"""Tests del modelo congelado: backbone/FPN/decoder/anclas/cabezas y LaneTR completo.

    python tests/test_models.py        (requiere torch)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lanetr.models.anchors import LaneAnchors
from lanetr.models.backbone import build_backbone
from lanetr.models.decoder import LaneDecoder
from lanetr.models.deform_attn import MSDeformAttn
from lanetr.models.fpn import FPN
from lanetr.models.head import LaneHead, decode_lanes
from lanetr.models.lanetr import LaneTR

B, NQ, L, R, D = 2, 12, 6, 144, 256


def test_backbone_fpn_channels():
    bb = build_backbone("dla34", pretrained=False)
    assert len(bb.out_channels) == 3
    fpn = FPN(bb.out_channels, D)
    x = torch.randn(1, 3, 320, 800)
    feats = fpn(bb(x))
    assert len(feats) == 3 and all(f.shape[1] == D for f in feats)


def test_anchors_shapes():
    a = LaneAnchors(NQ, D)
    row_ys = torch.linspace(0, 319, R)
    assert a.prior_xs(row_ys, 320).shape == (NQ, R)
    assert a.ext_prior().shape == (NQ, 2)
    assert a.reference_points(0.5).shape == (NQ, 2)
    assert a.pos_embed().shape == (NQ, D)


def test_deform_attn_shape():
    attn = MSDeformAttn(D, n_levels=3, n_heads=8, n_points=4, n_ref_points=1)
    q = torch.randn(B, NQ, D)
    S = 40 * 100 + 20 * 50 + 10 * 25
    value = torch.randn(B, S, D)
    shapes = [(40, 100), (20, 50), (10, 25)]
    ref = torch.full((B, NQ, 3, 1, 2), 0.5)
    out = attn(q, ref, value, shapes)
    assert out.shape == (B, NQ, D)


def test_decoder_output():
    dec = LaneDecoder(D, num_layers=L, num_queries=NQ, num_levels=3, n_ref_points=1)
    feats = [torch.randn(B, D, 40, 100), torch.randn(B, D, 20, 50), torch.randn(B, D, 10, 25)]
    ref = torch.full((NQ, 2), 0.5)
    hs = dec(feats, query_pos=torch.randn(NQ, D), reference_points=ref)
    assert hs.shape == (L, B, NQ, D)


def test_head_shapes_and_ranges():
    head = LaneHead(D, R).eval()
    with torch.no_grad():
        pred = head(torch.randn(L, B, NQ, D))
    assert pred["conf"].shape == (L, B, NQ)
    assert pred["xs"].shape == (L, B, NQ, R)
    for key in ("start_y", "length"):
        assert pred[key].min() >= 0.0 and pred[key].max() <= 1.0


def test_lanetr_forward_shapes():
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=L, num_rows=R).eval()
    x = torch.randn(B, 3, 320, 800)
    with torch.no_grad():
        pred = model(x)
    assert pred["conf"].shape == (L, B, NQ)
    assert pred["xs"].shape == (L, B, NQ, R)


def test_lanetr_predict_common_format_and_cap():
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=L, num_rows=R)
    x = torch.randn(2, 3, 320, 800)
    out = model.predict(x, src_sizes=[(1640, 590), (1280, 720)], conf_thresh=None)
    assert len(out) == 2
    for item in out:
        assert "Lines" in item and "Scores" in item
        assert len(item["Lines"]) <= 4                       # tope de carriles
        assert len(item["Lines"]) == len(item["Scores"])


def test_lanetr_grad_flows():
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=L, num_rows=R)
    x = torch.randn(1, 3, 320, 800)
    pred = model(x)
    loss = pred["conf"].mean() + pred["xs"].mean() + pred["start_y"].mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_return_attn_runs():
    model = LaneTR(pretrained=False, num_queries=NQ, num_layers=L, num_rows=R).eval()
    x = torch.randn(1, 3, 320, 800)
    with torch.no_grad():
        pred, info = model(x, return_attn=True)
    assert "attn" in info and "shapes" in info
    lanes = decode_lanes(pred, conf_thresh=None, num_rows=R)
    assert len(lanes) == 1


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
