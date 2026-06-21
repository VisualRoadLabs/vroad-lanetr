"""Transformaciones geométricas y de color, comunes a cualquier resolución de entrada.

Todas operan sobre un `sample` (dict) y transforman **a la vez** la imagen y los puntos de los
carriles, para que sigan alineados. Pipeline típico:

    train: CropResize -> RandomHorizontalFlip -> RandomAffine -> Normalize [-> EncodeTargets]
    val:   CropResize -> Normalize [-> EncodeTargets]

`CropResize` es la **única** transformación consciente de la resolución: recibe la imagen en su
resolución NATIVA (cualquiera: 1640×590, 1280×720, …) y la lleva al espacio del modelo (800×320)
recortando la franja de cielo (`crop_top_ratio`) y redimensionando. A partir de ahí todo vive en
800×320 y el resto de transformaciones son agnósticas a la resolución de origen.

`sample` tiene las claves:
    image      : PIL.Image (RGB)  ->  tras Normalize pasa a torch.FloatTensor (3,H,W)
    lanes      : list[np.ndarray (N,2) float32]   puntos (x,y) en el espacio actual
    slots      : list[int]                  (opcional; el formato común no trae slot)
    existence  : tuple[int,...] | None      (opcional)
    meta       : dict
"""
from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from . import target_encoding as TE
from .format import CROP_TOP_RATIO, source_to_model

# Estadísticas de ImageNet (el backbone se preentrena con ellas).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_BILINEAR = Image.Resampling.BILINEAR


def _affine_matrix(angle_deg, scale, tx, ty, cx, cy) -> np.ndarray:
    """Matriz 2x3 que rota+escala alrededor de (cx,cy) y traslada (tx,ty).
    Mapea punto ORIGINAL -> punto NUEVO."""
    a = math.radians(angle_deg)
    cos, sin = math.cos(a) * scale, math.sin(a) * scale
    return np.array([
        [cos, -sin, cx + tx - cos * cx + sin * cy],
        [sin,  cos, cy + ty - sin * cx - cos * cy],
    ], dtype=np.float64)


def _apply_matrix_to_lanes(lanes, M):
    out = []
    for pts in lanes:
        if len(pts) == 0:
            out.append(pts)
            continue
        homog = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], axis=1)  # (N,3)
        out.append((homog @ M.T).astype(np.float32))  # (N,2)
    return out

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample, rng):
        for t in self.transforms:
            sample = t(sample, rng)
        return sample


class CropResize:
    """Lleva la imagen de su resolución NATIVA al espacio del modelo (img_w×img_h).

    Recorta la franja superior (cielo) — una **fracción** `crop_top_ratio` de la altura, así vale
    para cualquier resolución — y redimensiona el resto a (img_w, img_h). Reusa exactamente el
    mismo mapeo que `lanetr.data.format`, de modo que `predict()` puede invertirlo.
    """

    def __init__(self, img_w=800, img_h=320, crop_top_ratio=CROP_TOP_RATIO):
        self.img_w, self.img_h, self.crop_top_ratio = img_w, img_h, crop_top_ratio

    def __call__(self, sample, rng):
        img = sample["image"]
        W, H = img.size
        crop = int(round(self.crop_top_ratio * H))
        img = img.crop((0, crop, W, H)).resize((self.img_w, self.img_h), _BILINEAR)
        sample["image"] = img
        sample["lanes"] = [source_to_model(pts, W, H, self.crop_top_ratio, self.img_w, self.img_h)
                           for pts in sample["lanes"]]
        sample["meta"]["src_size"] = (W, H)
        sample["meta"]["crop_top_ratio"] = self.crop_top_ratio
        sample["meta"]["img_size"] = (self.img_w, self.img_h)
        return sample


class RandomHorizontalFlip:
    """Espejo horizontal: invierte x, el orden de carriles y (si existen) los flags de existencia."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample, rng):
        if rng.random() >= self.p:
            return sample
        W, _ = sample["image"].size
        sample["image"] = sample["image"].transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        lanes = []
        for pts in sample["lanes"]:
            q = pts.copy()
            q[:, 0] = (W - 1) - q[:, 0]
            lanes.append(q)

        # reordenar de izquierda a derecha por la x media e invertir la existencia (si la hay)
        order = sorted(range(len(lanes)), key=lambda i: float(lanes[i][:, 0].mean()) if len(lanes[i]) else 0.0)
        sample["lanes"] = [lanes[i] for i in order]
        existence = sample.get("existence")
        if existence is not None:
            flipped = tuple(existence[::-1])
            sample["existence"] = flipped
            sample["slots"] = [i for i, f in enumerate(flipped) if f == 1]
        elif sample.get("slots") is not None:
            sample["slots"] = [sample["slots"][i] for i in order]
        return sample


class RandomAffine:
    """Rotación/escala/traslación aleatoria en el espacio de la imagen final."""

    def __init__(self, degrees=6.0, scale=(0.9, 1.1), translate=(0.05, 0.05), p=0.5):
        self.degrees, self.scale, self.translate, self.p = degrees, scale, translate, p

    def __call__(self, sample, rng):
        if rng.random() >= self.p:
            return sample
        W, H = sample["image"].size
        angle = rng.uniform(-self.degrees, self.degrees)
        s = rng.uniform(self.scale[0], self.scale[1])
        tx = rng.uniform(-self.translate[0], self.translate[0]) * W
        ty = rng.uniform(-self.translate[1], self.translate[1]) * H

        M = _affine_matrix(angle, s, tx, ty, W / 2.0, H / 2.0)
        M3 = np.vstack([M, [0, 0, 1]])
        inv = np.linalg.inv(M3)
        data = (inv[0, 0], inv[0, 1], inv[0, 2], inv[1, 0], inv[1, 1], inv[1, 2])
        sample["image"] = sample["image"].transform((W, H), Image.Transform.AFFINE,
                                                     data, resample=_BILINEAR)
        sample["lanes"] = _apply_matrix_to_lanes(sample["lanes"], M)
        return sample


class Photometric:
    """Jitter de brillo y contraste (ayuda en Night/Dazzle/Shadow). No toca los carriles."""

    def __init__(self, brightness=0.0, contrast=0.0):
        self.brightness, self.contrast = brightness, contrast

    def __call__(self, sample, rng):
        if self.brightness <= 0 and self.contrast <= 0:
            return sample
        from PIL import ImageEnhance
        img = sample["image"]
        if self.brightness > 0:
            f = 1.0 + rng.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(f)
        if self.contrast > 0:
            f = 1.0 + rng.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(f)
        sample["image"] = img
        return sample


class Normalize:
    """Imagen PIL -> tensor (3,H,W) normalizado con estadísticas de ImageNet."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean, self.std = mean, std

    def __call__(self, sample, rng):
        arr = np.asarray(sample["image"], dtype=np.float32) / 255.0  # (H,W,3)
        arr = (arr - self.mean) / self.std
        sample["image"] = torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()
        return sample


class EncodeTargets:
    """Añade `sample['targets']`: la representación de filas-ancla que consume el modelo.

    Se ejecuta al final (usa `sample['lanes']`, arrays numpy ya en el espacio 800×320).
    """

    def __init__(self, num_rows=TE.ROWS_DEFAULT, img_w=800, img_h=320):
        self.row_ys = TE.make_row_ys(img_h, num_rows)
        self.img_w, self.img_h = img_w, img_h

    def __call__(self, sample, rng):
        sample["targets"] = TE.encode_sample(sample["lanes"], sample.get("slots"),
                                             self.row_ys, self.img_w, self.img_h)
        return sample


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """Tensor normalizado (3,H,W) -> array uint8 (H,W,3) para visualizar."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * std + mean
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def build_transforms(split, img_w=800, img_h=320, crop_top_ratio=CROP_TOP_RATIO, augment=None,
                     encode_targets=False, num_rows=TE.ROWS_DEFAULT,
                     hflip_prob=0.5, rotation_deg=0.0, scale_jitter=0.0,
                     brightness=0.0, contrast=0.0):
    """Pipeline por split, controlado por las perillas `data.aug.*` (§42-bis).

    `augment` por defecto = True solo en 'train'. Cada augmentación se activa solo si su perilla
    es > 0 (los defaults del schema dejan solo el flip): hflip (`hflip_prob`), afín
    (`rotation_deg`, `scale_jitter`) y fotométrico (`brightness`, `contrast`).
    """
    if augment is None:
        augment = (split == "train")
    ts = [CropResize(img_w, img_h, crop_top_ratio)]
    if augment:
        if hflip_prob > 0:
            ts.append(RandomHorizontalFlip(p=hflip_prob))
        if rotation_deg > 0 or scale_jitter > 0:
            ts.append(RandomAffine(degrees=rotation_deg,
                                   scale=(1.0 - scale_jitter, 1.0 + scale_jitter),
                                   translate=(0.0, 0.0), p=1.0))
        if brightness > 0 or contrast > 0:
            ts.append(Photometric(brightness=brightness, contrast=contrast))
    ts += [Normalize()]
    if encode_targets:
        ts += [EncodeTargets(num_rows=num_rows, img_w=img_w, img_h=img_h)]
    return Compose(ts)
