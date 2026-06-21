"""Piezas declarativas del contrato que viajan CON el modelo.

El dashboard de ML Training NO hardcodea nada del modelo: **renderiza lo que el modelo declara**.
Estas tres estructuras son ligeras (no importan torch) y la plataforma las lee tal cual:

- `MODEL_INFO`    — identidad y forma de E/S del modelo.
- `CONFIG_SCHEMA` — un parámetro por entrada del formulario "crear entrenamiento", agrupado en
                    `arch | optim | schedule | loss | data` (§42-bis). El `arch` barre la
                    **capacidad de la misma familia** (nº de queries, profundidad, puntos de
                    referencia, refinamiento). Lo verdaderamente congelado (backbone de familia,
                    quitar el transformer, hidden_dim/FFN, niveles FPN, cabeza, input_size) NO
                    aparece. **No hay** `geo_metric` (siempre LaneIoU) ni grupo `sampler` (el
                    énfasis en curvas es un experimento que no vive en el modelo congelado).
- `METRICS_SPEC`  — etiquetas/orden y sentido ("más alto = mejor") de los escalares logueados.

`DEFAULT_CONFIG` es la config base efectiva (la misma que `configs/lanetr_culane.yaml`), de la que
`CONFIG_SCHEMA` toma los `default` de cada campo.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lanetr")     # fuente de verdad = pyproject.toml
except PackageNotFoundError:            # ejecutado desde el árbol sin instalar
    __version__ = "0.0.0"

# --------------------------------------------------------------------------- #
# Identidad y E/S del modelo
# --------------------------------------------------------------------------- #
MODEL_INFO = {
    "name": "lanetr",
    "version": __version__,
    "arch": "DETR deformable + FPN(DLA-34) + LaneIoU + matching húngaro",
    "input": (3, 320, 800),          # (C, H, W) — la imagen entra SIEMPRE a 800×320
    "img_size": (800, 320),          # (W, H) espacio del modelo
    "num_rows": 144,                 # filas-ancla por carril
    "num_queries": 12,               # carriles candidatos (default; barrible en `arch`)
    "num_decoder_layers": 6,         # profundidad del decoder (default; barrible)
    "n_ref_points": 4,               # puntos de referencia por query (default; barrible)
    "ref_refine": "mlp",             # refinamiento iterativo de referencias (default; barrible)
    "d_model": 256,
    "backbone": "dla34",
    "max_lanes": 4,                  # tope de carriles emitidos (= máximo de CULane)
    "params_m": 25.1,                # millones de parámetros (nominal, defaults; ver count_parameters)
    "common_format": ".lines.json",  # formato de salida de predict()
}


# --------------------------------------------------------------------------- #
# Config base efectiva (= configs/lanetr_culane.yaml). `arch` barre capacidad de la familia;
# los campos congelados (backbone, d_model, dim_ff, nhead, n_points, img_*) viven aquí pero NO
# se exponen en el formulario.
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG: dict = {
    "name": "lanetr_dla34_culane",
    "arch": {
        # --- barribles (en CONFIG_SCHEMA) ---
        "num_queries": 12, "num_decoder_layers": 6, "n_ref_points": 4,
        "ref_refine": "mlp", "load_strict": True,
        # --- congelados (no en el formulario) ---
        "backbone": "dla34", "pretrained": True, "d_model": 256, "nhead": 8, "dim_ff": 1024,
        "n_points": 4, "num_rows": 144, "img_w": 800, "img_h": 320,
        "ref_y_top": 0.15, "ref_y_bottom": 0.95,
    },
    "optim": {"lr": 3.0e-4, "backbone_lr_mult": 0.1, "weight_decay": 1.0e-4,
              "grad_clip": 0.1, "ema_decay": 0.9999, "slow_mult": 0.1},
    "schedule": {"epochs": 50, "warmup_epochs": 3, "scheduler": "cosine", "min_lr": 1.0e-6},
    "loss": {"w_cls": 1.0, "w_iou": 2.0, "w_xy": 0.5, "w_ext": 0.5, "focal_gamma": 2.0,
             "cost_cls": 1.0, "cost_iou": 2.0, "cost_xy": 0.5, "cost_ext": 0.5,
             "aux_loss": True, "focal_alpha": 0.25, "w_theta": 0.0, "w_smooth": 0.0},
    "data": {"dataset_manifest": "culane@1", "batch_size": 32, "num_workers": 8, "seed": 42,
             "crop_top_ratio": 270.0 / 590.0,
             "aug": {"hflip_prob": 0.5, "rotation_deg": 0.0, "scale_jitter": 0.0,
                     "brightness": 0.0, "contrast": 0.0}},
    "train": {"amp": True, "channels_last": True, "freeze_bn": True, "eval_conf_thresh": 0.5},
}


# --------------------------------------------------------------------------- #
# Formulario "crear entrenamiento": un campo por hiperparámetro barrible (§42-bis).
# --------------------------------------------------------------------------- #
CONFIG_SCHEMA = [
    # --- arch: capacidad de la misma familia (afecta a la forma de los pesos) ---
    {"path": "arch.num_queries", "type": "choice", "choices": [4, 12, 20], "default": 12,
     "group": "arch", "label": "Nº queries", "help": "Carriles candidatos (anchors/consultas)."},
    {"path": "arch.num_decoder_layers", "type": "int", "default": 6, "min": 1, "max": 6,
     "group": "arch", "label": "Capas decoder", "help": "Profundidad del decoder deformable."},
    {"path": "arch.n_ref_points", "type": "int", "default": 4, "min": 1, "max": 4,
     "group": "arch", "label": "Puntos de referencia",
     "help": "Puntos de la atención deformable a lo largo del carril (1 = centro)."},
    {"path": "arch.ref_refine", "type": "choice", "choices": ["xs", "mlp"], "default": "mlp",
     "group": "arch", "label": "Refinamiento de refs",
     "help": "Refinamiento iterativo de referencias: mlp (DAB-DETR) o xs (deriva del xs predicho)."},
    {"path": "arch.load_strict", "type": "bool", "default": True,
     "group": "arch", "label": "Carga estricta (FT)",
     "help": "En fine-tuning, exige forma idéntica del checkpoint; si no, carga parcial."},
    # --- optim ---
    {"path": "optim.lr", "type": "float", "default": 3.0e-4, "min": 1e-4, "max": 1e-3,
     "group": "optim", "label": "Learning rate", "help": "LR base de AdamW (transformer/cabezas)."},
    {"path": "optim.backbone_lr_mult", "type": "float", "default": 0.1, "min": 0.05, "max": 0.5,
     "group": "optim", "label": "Backbone LR ×", "help": "Multiplicador del LR del backbone."},
    {"path": "optim.weight_decay", "type": "float", "default": 1.0e-4, "min": 1e-4, "max": 1e-2,
     "group": "optim", "label": "Weight decay", "help": "Regularización de AdamW."},
    {"path": "optim.grad_clip", "type": "float", "default": 0.1, "min": 0.05, "max": 1.0,
     "group": "optim", "label": "Grad clip", "help": "Recorte de norma del gradiente."},
    {"path": "optim.ema_decay", "type": "float", "default": 0.9999, "min": 0.999, "max": 0.9999,
     "group": "optim", "label": "EMA decay", "help": "Momentum del EMA (= 1 − momentum)."},
    # --- schedule ---
    {"path": "schedule.epochs", "type": "int", "default": 50, "min": 15, "max": 50,
     "group": "schedule", "label": "Épocas", "help": "Épocas totales (15 ablation, 50 final)."},
    {"path": "schedule.warmup_epochs", "type": "int", "default": 3, "min": 1, "max": 5,
     "group": "schedule", "label": "Warmup épocas", "help": "Calentamiento lineal del LR."},
    {"path": "schedule.scheduler", "type": "choice", "choices": ["cosine", "step", "poly"],
     "default": "cosine", "group": "schedule", "label": "Scheduler", "help": "Tipo de decaimiento."},
    {"path": "schedule.min_lr", "type": "float", "default": 1.0e-6, "min": 1e-6, "max": 1e-5,
     "group": "schedule", "label": "Min LR", "help": "LR final del decaimiento."},
    # --- loss (los pesos = términos del criterion; cost_* = costes del matcher) ---
    {"path": "loss.w_cls", "type": "float", "default": 1.0, "min": 0.5, "max": 3.0,
     "group": "loss", "label": "w_cls", "help": "Peso de la clasificación focal."},
    {"path": "loss.w_iou", "type": "float", "default": 2.0, "min": 1.0, "max": 5.0,
     "group": "loss", "label": "w_iou", "help": "Peso de la LaneIoU (ajuste de la línea)."},
    {"path": "loss.w_xy", "type": "float", "default": 0.5, "min": 0.1, "max": 2.0,
     "group": "loss", "label": "w_xy", "help": "Peso del L1 lateral (x) fila a fila."},
    {"path": "loss.w_ext", "type": "float", "default": 0.5, "min": 0.1, "max": 2.0,
     "group": "loss", "label": "w_ext", "help": "Peso de longitud/extremos (start_y, length)."},
    {"path": "loss.focal_gamma", "type": "float", "default": 2.0, "min": 1.0, "max": 3.0,
     "group": "loss", "label": "focal γ", "help": "Foco de la focal loss."},
    {"path": "loss.cost_cls", "type": "float", "default": 1.0, "min": 0.0, "max": 3.0,
     "group": "loss", "label": "cost_cls", "help": "Coste de clasificación del matcher húngaro."},
    {"path": "loss.cost_iou", "type": "float", "default": 2.0, "min": 0.0, "max": 5.0,
     "group": "loss", "label": "cost_iou", "help": "Coste LaneIoU del matcher húngaro."},
    {"path": "loss.cost_xy", "type": "float", "default": 0.5, "min": 0.0, "max": 2.0,
     "group": "loss", "label": "cost_xy", "help": "Coste lateral del matcher húngaro."},
    {"path": "loss.aux_loss", "type": "bool", "default": True,
     "group": "loss", "label": "Aux loss", "help": "Pérdidas auxiliares en capas intermedias."},
    # --- data ---
    {"path": "data.dataset_manifest", "type": "str", "default": "culane@1",
     "group": "data", "label": "Dataset (manifiesto)", "help": "Qué dataset/mezcla (id@versión)."},
    {"path": "data.batch_size", "type": "int", "default": 32, "min": 16, "max": 32,
     "group": "data", "label": "Batch size", "help": "Tamaño de batch (A100/L4 según el tier)."},
    {"path": "data.num_workers", "type": "int", "default": 8, "min": 4, "max": 16,
     "group": "data", "label": "Num workers", "help": "Hilos del dataloader."},
    {"path": "data.seed", "type": "int", "default": 42, "min": 0, "max": 999999,
     "group": "data", "label": "Seed", "help": "Semilla (varianza/reproducibilidad)."},
    {"path": "data.aug.hflip_prob", "type": "float", "default": 0.5, "min": 0.0, "max": 0.5,
     "group": "data", "label": "Flip H prob", "help": "Probabilidad de volteo horizontal."},
    {"path": "data.aug.rotation_deg", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0,
     "group": "data", "label": "Rotación (°)", "help": "Rotación aleatoria (ayuda a curvas)."},
    {"path": "data.aug.scale_jitter", "type": "float", "default": 0.0, "min": 0.0, "max": 0.2,
     "group": "data", "label": "Scale jitter", "help": "Escala aleatoria ±."},
    {"path": "data.aug.brightness", "type": "float", "default": 0.0, "min": 0.0, "max": 0.4,
     "group": "data", "label": "Brillo", "help": "Jitter de brillo (Night/Dazzle/Shadow)."},
    {"path": "data.aug.contrast", "type": "float", "default": 0.0, "min": 0.0, "max": 0.4,
     "group": "data", "label": "Contraste", "help": "Jitter de contraste (Night/Dazzle/Shadow)."},
]


# --------------------------------------------------------------------------- #
# Etiquetas/orden de las métricas escalares (opcional; el dashboard plotea todo lo logueado).
# --------------------------------------------------------------------------- #
METRICS_SPEC = {
    "f1/global":   {"label": "F1 global", "higher_is_better": True, "order": 0},
    "f1/curve":    {"label": "F1 curva",  "higher_is_better": True, "order": 1},
    "loss/total":  {"label": "Loss total", "higher_is_better": False, "order": 2},
    "loss/cls":    {"label": "Loss cls",   "higher_is_better": False, "order": 3},
    "loss/iou":    {"label": "Loss LaneIoU", "higher_is_better": False, "order": 4},
    "loss/xy":     {"label": "Loss xy",    "higher_is_better": False, "order": 5},
    "loss/ext":    {"label": "Loss ext",   "higher_is_better": False, "order": 6},
    "lr":          {"label": "LR",         "higher_is_better": None, "order": 7},
    "gpu/mem_mb":  {"label": "GPU mem (MB)", "higher_is_better": None, "order": 8},
    "gpu/util":    {"label": "GPU util",   "higher_is_better": None, "order": 9},
}
