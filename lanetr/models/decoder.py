"""Decoder transformer tipo DETR con atención DEFORMABLE.

Un conjunto FIJO de `num_queries` "fichas" (queries) que, capa a capa, se miran entre sí
(self-attention densa) y miran la pirámide de features del FPN (cross-attention DEFORMABLE),
produciendo un vector por query. Las cabezas convierten cada vector en (confianza + geometría).

Capacidad de la familia (§42-bis), barrible por config:
  - `num_layers` (1–6): profundidad del decoder.
  - `n_ref_points` (1–4): puntos de referencia por query. 1 = un punto a media altura; >1 los
    reparte A LO LARGO del carril para "ver" la curva.
  - `ref_refine` (xs/mlp): refinamiento iterativo de la x de las referencias hacia el carril
    predicho, capa a capa (estilo DAB-DETR / Sparse Laneformer). "mlp" = un MLP predice el delta;
    "xs" = deriva la x del propio `xs` que la cabeza ya predice.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .deform_attn import MSDeformAttn
from .head import MLP, _inverse_sigmoid


class DeformableDecoderLayer(nn.Module):
    """Capa de decoder con cross-attention DEFORMABLE (self-attn densa entre queries)."""

    def __init__(self, d_model: int = 256, nhead: int = 8, dim_ff: int = 1024,
                 dropout: float = 0.1, n_levels: int = 3, n_points: int = 4, n_ref_points: int = 1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformAttn(d_model, n_levels, nhead, n_points, n_ref_points)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _with_pos(t, pos):
        return t if pos is None else t + pos

    def forward(self, tgt, query_pos, reference_points, memory, spatial_shapes, return_sampling=False):
        q = k = self._with_pos(tgt, query_pos)
        sa, _ = self.self_attn(q, k, value=tgt, need_weights=False)
        tgt = self.norm1(tgt + self.dropout1(sa))
        res = self.cross_attn(self._with_pos(tgt, query_pos), reference_points, memory,
                              spatial_shapes, return_sampling=return_sampling)
        ca = res[0] if return_sampling else res
        tgt = self.norm2(tgt + self.dropout2(ca))
        ff = self.linear2(self.dropout(self.act(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ff))
        if return_sampling:
            return tgt, res[1], res[2]   # tgt, sampling_locations, attn
        return tgt


class LaneDecoder(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 6,
                 num_queries: int = 12, dim_ff: int = 1024, dropout: float = 0.1,
                 num_levels: int = 3, n_points: int = 4, n_ref_points: int = 1,
                 ref_refine: bool = False, ref_refine_mode: str = "mlp"):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.n_ref_points = n_ref_points
        self.ref_refine = ref_refine
        self.ref_refine_mode = ref_refine_mode      # "mlp" (DAB-DETR) | "xs" (deriva del xs de la cabeza)
        self.query_embed = nn.Embedding(num_queries, d_model)     # pos. de las queries (fallback)
        self.layers = nn.ModuleList(
            [DeformableDecoderLayer(d_model, nhead, dim_ff, dropout, num_levels, n_points, n_ref_points)
             for _ in range(num_layers)])
        if ref_refine and ref_refine_mode == "mlp":
            # refinamiento iterativo (estilo DAB-DETR): tras cada capa, mueve la x de los puntos de
            # referencia hacia el carril predicho. Init a 0 -> delta=0 -> arrancan en la recta del ancla.
            self.ref_refine_mlp = MLP(d_model, d_model, n_ref_points)
            nn.init.zeros_(self.ref_refine_mlp.layers[-1].weight)
            nn.init.zeros_(self.ref_refine_mlp.layers[-1].bias)
        self.norm = nn.LayerNorm(d_model)

    def _build_memory(self, feats):
        # La atención deformable NO usa codificación posicional ni embedding de nivel: la posición
        # llega por los puntos de referencia y el muestreo por nivel. Solo aplana las features.
        srcs, shapes = [], []
        for f in feats:
            shapes.append((f.shape[-2], f.shape[-1]))
            srcs.append(f.flatten(2).transpose(1, 2))             # (b,hw,c)
        return torch.cat(srcs, dim=1), shapes

    def forward(self, feats, query_pos, reference_points, ref_ys=None, ref_predict=None,
                need_attn: bool = False):
        """feats: lista de mapas del FPN. query_pos: (NQ,d) o (b,NQ,d).
        reference_points: (NQ,n_ref,2) — puntos de referencia (de las anclas). ref_ys: (n_ref,)
        alturas fijas de las referencias (para el refinamiento). ref_predict: callable opcional
        (modo "xs") que da la x del carril predicho en `ref_ys`."""
        b = feats[0].shape[0]
        memory, shapes = self._build_memory(feats)
        if query_pos is None:
            query_pos = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (b,N,d)
        elif query_pos.dim() == 2:
            query_pos = query_pos.unsqueeze(0).expand(b, -1, -1)  # (NQ,d) -> (b,NQ,d)
        tgt = torch.zeros_like(query_pos)

        Rf = self.n_ref_points
        if reference_points is None:                              # por defecto: centro
            reference_points = torch.full((self.num_queries, Rf, 2), 0.5, device=memory.device)
        if reference_points.dim() == 2:                           # (NQ,2) -> (NQ,1,2)
            reference_points = reference_points.unsqueeze(1)
        if reference_points.dim() == 3:                           # (NQ,n_ref,2) -> (b,NQ,n_ref,2)
            reference_points = reference_points.unsqueeze(0).expand(b, -1, -1, -1)
        cur = reference_points                                    # (b,NQ,n_ref,2): refs de la capa actual
        if ref_ys is not None:
            ref_ys = ref_ys.to(memory.device)

        outs, samps = [], []
        for li, layer in enumerate(self.layers):
            # (b,NQ,n_ref,2) -> (b,NQ,n_levels,n_ref,2): mismas refs (normalizadas) por nivel
            ref = cur[:, :, None, :, :].expand(-1, -1, self.num_levels, -1, -1)
            res = layer(tgt, query_pos, ref, memory, shapes, return_sampling=need_attn)
            tgt = res[0] if need_attn else res
            hs_l = self.norm(tgt)
            outs.append(hs_l)
            if need_attn:
                samps.append((res[1], res[2]))                    # (sampling_locations, attn)
            # refinamiento iterativo: mueve la x de las refs hacia el carril predicho
            if self.ref_refine and li < self.num_layers - 1:
                if self.ref_refine_mode == "xs" and ref_predict is not None:
                    new_x = ref_predict(hs_l).detach().clamp(1e-4, 1 - 1e-4)   # (b,NQ,n_ref)
                else:                                                          # MLP (DAB-DETR)
                    dx = self.ref_refine_mlp(hs_l)                            # (b,NQ,n_ref)
                    new_x = torch.sigmoid(_inverse_sigmoid(cur[..., 0].detach()) + dx)
                new_y = (ref_ys.view(1, 1, Rf).expand_as(new_x)
                         if ref_ys is not None else cur[..., 1].detach())
                cur = torch.stack([new_x, new_y], dim=-1)         # refs para la capa siguiente
        hs = torch.stack(outs, dim=0)                            # (num_layers, b, num_queries, d)
        return (hs, samps, shapes) if need_attn else hs
