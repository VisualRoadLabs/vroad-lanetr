"""Exporta los artefactos declarativos del contrato a ficheros JSON.

La CI sube estos JSON al bucket de datasets (`_model/<modelo>@<sha>/`) para que el **dashboard**
de ML Training pueda renderizar el formulario y las métricas SIN importar torch (CLAUDE.md §43):

    config_schema.json   -> CONFIG_SCHEMA   (formulario "crear entrenamiento")
    model_info.json      -> MODEL_INFO       (identidad y E/S)
    metrics_spec.json    -> METRICS_SPEC     (etiquetas/orden de escalares)

Uso:  python tools/export_contract.py --out ./_contract
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lanetr.contract.spec import CONFIG_SCHEMA, METRICS_SPEC, MODEL_INFO  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./_contract", help="directorio de salida")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "config_schema.json": CONFIG_SCHEMA,
        "model_info.json": {**MODEL_INFO, "input": list(MODEL_INFO["input"]),
                            "img_size": list(MODEL_INFO["img_size"])},
        "metrics_spec.json": METRICS_SPEC,
    }
    for name, obj in artifacts.items():
        (out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"escrito {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
