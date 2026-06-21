"""Modelo completo LaneTR (familia congelada): backbone + FPN + decoder deformable + cabezas.

    imagen (B,3,320,800)
      -> backbone (DLA-34)            -> [C3,C4,C5]
      -> FPN                          -> [P3,P4,P5] (256 ch)
      -> decoder transformer deform.  -> hs (L, B, num_queries, 256)
      -> cabezas                      -> {conf, xs, start_y, length, theta}

Arquitectura de la MISMA FAMILIA, con capacidad barrible (§42-bis): `num_queries`, `num_layers`,
`n_ref_points` (1–4, puntos de referencia a lo largo del carril) y `ref_refine` (xs/mlp,
refinamiento iterativo de referencias). Lo CONGELADO (→ Sandbox) es el backbone de familia,
quitar el transformer, hidden_dim/FFN, niveles del FPN, la cabeza, la codificación posicional y
el input_size 320×800.

`forward` devuelve el dict de predicciones de TODAS las capas (para pérdidas auxiliares).
`predict` decodifica la última capa y emite carriles en **formato común** `.lines.json`, ya
mapeados a la resolución de origen de cada imagen.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.format import CROP_TOP_RATIO, lanes_to_common, model_to_source
from ..data.target_encoding import make_row_ys
from .anchors import LaneAnchors
from .backbone import build_backbone
from .decoder import LaneDecoder
from .fpn import FPN
from .head import LaneHead, decode_lanes


class LaneTR(nn.Module):
    def __init__(self, backbone: str = "dla34", pretrained: bool = True, d_model: int = 256,
                 num_queries: int = 12, num_layers: int = 6, num_rows: int = 144,
                 nhead: int = 8, dim_ff: int = 1024, img_w: int = 800, img_h: int = 320,
                 n_points: int = 4, n_ref_points: int = 4, ref_refine: bool = True,
                 ref_refine_mode: str = "mlp", ref_y_top: float = 0.15, ref_y_bottom: float = 0.95):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained)
        self.fpn = FPN(self.backbone.out_channels, d_model)
        self.decoder = LaneDecoder(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                   num_queries=num_queries, dim_ff=dim_ff,
                                   num_levels=len(self.backbone.out_channels),
                                   n_points=n_points, n_ref_points=n_ref_points,
                                   ref_refine=ref_refine, ref_refine_mode=ref_refine_mode)
        self.head = LaneHead(d_model, num_rows, residual_xs=True)
        self.anchors = LaneAnchors(num_queries, d_model)
        self.num_rows = num_rows
        self.num_queries = num_queries
        self.img_w = img_w
        self.img_h = img_h
        self.n_ref_points = n_ref_points
        self.ref_refine = ref_refine
        self.ref_refine_mode = ref_refine_mode
        self.ref_y_top = ref_y_top
        self.ref_y_bottom = ref_y_bottom
        # atención deformable y prior posicional: marcados para LR lento en el optimizador.
        self.register_buffer("row_ys", torch.tensor(make_row_ys(img_h, num_rows)))

    def _make_ref_predict(self, ref_ys):
        """Callable hs_l -> x del carril PREDICHO en las alturas `ref_ys` (b,NQ,n_ref).
        Reusa el xs que la cabeza ya predice (prior + delta), leído en las filas más cercanas a
        `ref_ys`. Es la señal del modo de refinamiento "xs" (supervisada por la pérdida)."""
        norm_rows = (self.row_ys.to(ref_ys.device) / (self.img_h - 1)).float()    # (num_rows,) en [0,1]
        ref_row_idx = (norm_rows[None, :] - ref_ys[:, None]).abs().argmin(dim=1)   # (n_ref,)
        prior_xs = self.anchors.prior_xs(self.row_ys, self.img_h)                  # (NQ, num_rows)

        def ref_predict(hs_l):                                    # hs_l: (b,NQ,d)
            xs_full = prior_xs.unsqueeze(0) + self.head.xs(hs_l)  # (b,NQ,num_rows) = xs de la cabeza
            return xs_full[..., ref_row_idx]                      # (b,NQ,n_ref)
        return ref_predict

    def forward(self, images: torch.Tensor, return_attn: bool = False):
        feats = self.fpn(self.backbone(images))
        # puntos de referencia a lo largo del carril (n_ref_points); 1 = centro a media altura.
        ref = self.anchors.reference_points_multi(self.n_ref_points, self.ref_y_top, self.ref_y_bottom)
        ref_ys = self.anchors.ref_heights(self.n_ref_points, self.ref_y_top, self.ref_y_bottom)
        ref_predict = self._make_ref_predict(ref_ys) if (self.ref_refine and self.ref_refine_mode == "xs") else None
        dec = self.decoder(feats, query_pos=self.anchors.pos_embed(), reference_points=ref,
                           ref_ys=ref_ys, ref_predict=ref_predict, need_attn=return_attn)
        prior = self.anchors.prior_xs(self.row_ys, self.img_h)    # (NQ,R)
        prior_ext = self.anchors.ext_prior()
        if return_attn:
            hs, attn, shapes = dec
            pred = self.head(hs, prior_xs=prior, prior_ext=prior_ext)
            return pred, {"attn": attn, "shapes": shapes}
        return self.head(dec, prior_xs=prior, prior_ext=prior_ext)

    @torch.no_grad()
    def predict(self, images: torch.Tensor, src_sizes: list[tuple[int, int]] | None = None,
                conf_thresh: float | None = 0.5, crop_top_ratio: float = CROP_TOP_RATIO,
                max_lanes: int = 4, timestamps: list[int] | None = None) -> list[dict]:
        """Inferencia -> lista (por imagen) de dicts en **formato común** (§2): `Lines` + `Scores`.

        El modelo decodifica en su espacio 800×320 y, si se pasan `src_sizes=[(W,H), ...]` (la
        resolución NATIVA de cada imagen: 1640×590 CULane, 1280×720 usuario, …), mapea los puntos
        de vuelta a esa resolución de origen. Si no se pasan, los deja en el espacio del modelo.
        `Scores` lleva la confianza por carril (sigmoide del logit), alineada con `Lines`. Se
        emiten como mucho `max_lanes` carriles (los de mayor confianza), el tope de CULane.
        """
        self.eval()
        pred = self.forward(images)
        per_img = decode_lanes(pred, layer=-1, conf_thresh=conf_thresh, num_rows=self.num_rows,
                               img_w=self.img_w, img_h=self.img_h)
        B = len(per_img)
        timestamps = timestamps if timestamps is not None else [None] * B
        out = []
        for b in range(B):
            lanes = sorted(per_img[b], key=lambda d: -d["conf"])[:max_lanes]
            scores = [lane["conf"] for lane in lanes]
            if src_sizes is not None:                             # mapear a la resolución de origen
                sw, sh = src_sizes[b]
                pts = [model_to_source(lane["points"], sw, sh, crop_top_ratio,
                                       self.img_w, self.img_h) for lane in lanes]
            else:                                                 # dejar en el espacio del modelo
                pts = [lane["points"] for lane in lanes]
            out.append(lanes_to_common(pts, scores, timestamps[b]))
        return out
