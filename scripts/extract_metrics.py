"""Calcula métricas por capa para un modelo y guarda un CSV en ``output.dir``.

Uso:
    python scripts/extract_metrics.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from layercollapse.data import load_prompts
from layercollapse.metrics import curvature, effective_rank, matrix_entropy
from layercollapse.models import hidden_states, load_model

METRICS = {
    "prompt_entropy": lambda Z, cfg: matrix_entropy(Z, alpha=cfg["metrics"]["alpha"]),
    "effective_rank": lambda Z, cfg: effective_rank(Z),
    "curvature": lambda Z, cfg: curvature(Z),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["runtime"]["seed"])

    m = cfg["model"]
    model, tok, dev = load_model(m["name"], m["revision"], m["dtype"], cfg["runtime"]["device"])
    d = cfg["data"]
    prompts = load_prompts(d["dataset"], d.get("subset"), d["split"], d["num_prompts"])

    rows = []
    for i, text in enumerate(tqdm(prompts, desc="prompts")):
        for layer, Z in enumerate(hidden_states(model, tok, text, dev, d["max_length"])):
            row = {"prompt": i, "layer": layer}
            for name in cfg["metrics"]["compute"]:
                row[name] = METRICS[name](Z, cfg)
            rows.append(row)

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics_per_prompt.csv", index=False)
    summary = df.drop(columns="prompt").groupby("layer").mean()
    summary.to_csv(out / "metrics_per_layer.csv")
    print(summary)


if __name__ == "__main__":
    main()
