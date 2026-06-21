"""Visualizadores que el modelo aporta al dashboard.

El `EpochVisualizer` de la plataforma es **ciego**: por cada `(nombre, fn)` de `VISUALIZERS` y por
cada imagen fija guarda `viz/epoch_XXX/<nombre>/imgK.png`. La única figura UNIVERSAL (`gt_vs_pred`)
la pone la plataforma; aquí van las **específicas del modelo**:

    anchors    — la línea-prior de cada ancla (usadas vs no) + barra de confianza por query.
    attention  — dónde mira cada query usada (puntos muestreados por la atención deformable).
    matcher    — coste del matching húngaro + asignación query→GT (si hay GT en el batch).

Contrato de cada `fn(model, batch, out) -> np.ndarray (H,W,3) uint8`:
  - `model` : el modelo (en eval).
  - `batch` : dict con `images` (B,3,H,W); opcional `rgb` (lista de uint8 HxWx3) y `targets`
              (lista de dicts estilo `encode_sample`: xs (G,R), valid (G,R), start (G,2), length).
  - `out`   : la salida de `model.forward` (dict de tensores (L,B,NQ,...)).
  - Renderiza la imagen índice 0 del batch y devuelve la figura como array RGB.
"""
from __future__ import annotations

import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..data.target_encoding import decode_lane, make_row_ys  # noqa: E402
from ..data.transforms import denormalize  # noqa: E402
from ..losses.matcher import HungarianMatcher  # noqa: E402


def _palette(n):
    return [colorsys.hsv_to_rgb(i / max(n, 1), 0.9, 1.0) for i in range(n)]


def _fig_to_array(fig) -> np.ndarray:
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    arr = buf[..., :3].copy()
    plt.close(fig)
    return arr


def _images(batch):
    """Devuelve el tensor de imágenes (B,3,H,W) tanto si `batch` es tensor como dict."""
    return batch if torch.is_tensor(batch) else batch["images"]


def _rgb(batch, i: int) -> np.ndarray:
    """Fondo RGB uint8 de la imagen i (de `batch['rgb']` o desnormalizando `images`)."""
    if not torch.is_tensor(batch) and batch.get("rgb") is not None:
        return np.asarray(batch["rgb"][i], dtype=np.uint8)
    return denormalize(_images(batch)[i])


def _forward(model, images, need_attn=False):
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        return model(images.to(dev), return_attn=need_attn)


# --------------------------------------------------------------------------- #
def anchors(model, batch, out, i: int = 0) -> np.ndarray:
    """Línea-prior de cada ancla (sólida=usada, punteada=no) + confianza por query."""
    rgb = _rgb(batch, i)
    img_w, img_h, thr = model.img_w, model.img_h, 0.5
    if out is None:
        out = _forward(model, _images(batch))
    conf = out["conf"][-1, i].sigmoid().cpu().numpy()
    used = conf >= thr
    prior = model.anchors.prior_xs(model.row_ys, img_h).detach().cpu().numpy()
    row_ys = make_row_ys(img_h, model.num_rows)
    NQ = prior.shape[0]
    pal = _palette(NQ)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
    axL.imshow(rgb); axL.axis("off")
    axL.set_title("Anclas (sólida=usada, punteada=no usada)", fontsize=9)
    for q in range(NQ):
        x = prior[q] * img_w
        axL.plot(x, row_ys, "-" if used[q] else ":", color=pal[q],
                 lw=2 if used[q] else 1, alpha=0.95 if used[q] else 0.35)
    axR.bar(range(NQ), conf, color=[pal[q] for q in range(NQ)])
    axR.axhline(thr, color="red", ls="--", lw=1, label=f"umbral {thr}")
    axR.set_xlabel("query"); axR.set_ylabel("confianza"); axR.set_ylim(0, 1)
    axR.set_title(f"{int(used.sum())}/{NQ} queries usadas", fontsize=9)
    axR.legend(fontsize=7)
    return _fig_to_array(fig)


def attention(model, batch, out, i: int = 0) -> np.ndarray:
    """Puntos que muestrea la atención deformable para las queries usadas (las más confiadas)."""
    images = _images(batch)
    rgb = _rgb(batch, i)
    pred, info = _forward(model, images, need_attn=True)
    conf = pred["conf"][-1, i].sigmoid().cpu().numpy()
    used = [q for q in range(model.num_queries) if conf[q] >= 0.5]
    if not used:
        used = list(np.argsort(-conf)[:4])
    show = sorted(used, key=lambda q: -conf[q])[:6]
    pal = _palette(len(show))
    sl = info["attn"][-1][0][i]  # última capa, imagen i: (NQ,n_heads,n_levels,Rf*P,2)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.imshow(rgb); ax.axis("off")
    for c, q in enumerate(show):
        pts = sl[q].reshape(-1, 2).cpu().numpy()
        ax.scatter(pts[:, 0] * model.img_w, pts[:, 1] * model.img_h, s=10,
                   color=pal[c], alpha=0.7, label=f"q{q}")
    ax.legend(fontsize=7, ncol=6, loc="upper center")
    ax.set_title("Atención deformable: dónde mira cada query usada", fontsize=10)
    return _fig_to_array(fig)


def matcher(model, batch, out, i: int = 0) -> np.ndarray:
    """Coste del matching húngaro + asignación query→GT (si el batch trae `targets`)."""
    rgb = _rgb(batch, i)
    img_w, img_h = model.img_w, model.img_h
    row_ys = make_row_ys(img_h, model.num_rows)
    if out is None:
        out = _forward(model, _images(batch))
    dev = next(model.parameters()).device

    targets = None if torch.is_tensor(batch) else batch.get("targets")
    gt = targets[i] if targets is not None and i < len(targets) else None
    if gt is None or np.asarray(gt["xs"]).shape[0] == 0:        # sin GT: solo dibuja el fondo
        fig, ax = plt.subplots(figsize=(6.5, 2.7))
        ax.imshow(rgb); ax.axis("off")
        ax.set_title("Matcher húngaro (sin GT en el batch)", fontsize=9)
        return _fig_to_array(fig)

    tgt = {"xs": torch.as_tensor(gt["xs"], dtype=torch.float32, device=dev),
           "valid": torch.as_tensor(gt["valid"], dtype=torch.bool, device=dev),
           "start_y": torch.as_tensor(np.asarray(gt["start"])[:, 1], dtype=torch.float32, device=dev),
           "length": torch.as_tensor(gt["length"], dtype=torch.float32, device=dev)}
    pred_b = {"conf": out["conf"][-1, i], "xs": out["xs"][-1, i],
              "start_y": out["start_y"][-1, i], "length": out["length"][-1, i]}
    mt = HungarianMatcher()
    comps = mt.cost_components(pred_b, tgt)
    q_idx, g_idx = mt.match_one(pred_b, tgt)
    total = comps["total"].detach().float().cpu().numpy()
    G = tgt["xs"].shape[0]
    pal = _palette(G)
    valid = np.asarray(gt["valid"])
    xs = out["xs"][-1, i].detach().cpu().numpy()
    fig, (axI, axM) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
    axI.imshow(rgb); axI.axis("off")
    axI.set_title("GT (blanco) y predicción emparejada (color)", fontsize=9)
    for qi, gi in zip(q_idx.tolist(), g_idx.tolist()):
        gl = decode_lane(np.asarray(gt["xs"])[gi], valid[gi], row_ys, img_w)
        if len(gl) >= 2:
            axI.plot(gl[:, 0], gl[:, 1], color="white", lw=4, alpha=0.85)
        pp = decode_lane(xs[qi], valid[gi], row_ys, img_w)
        if len(pp) >= 2:
            axI.plot(pp[:, 0], pp[:, 1], color=pal[gi], lw=2)
    im = axM.imshow(total, aspect="auto", cmap="viridis")
    axM.set_title("Coste húngaro (rojo=asignado)", fontsize=9)
    axM.set_xlabel("GT"); axM.set_ylabel("query"); axM.set_xticks(range(G))
    for qi, gi in zip(q_idx.tolist(), g_idx.tolist()):
        axM.add_patch(Rectangle((gi - 0.5, qi - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
    fig.colorbar(im, ax=axM, fraction=0.046)
    return _fig_to_array(fig)


VISUALIZERS = {
    "anchors": anchors,
    "attention": attention,
    "matcher": matcher,
}
