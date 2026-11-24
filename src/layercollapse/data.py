"""Carga de prompts para el análisis por capa."""

from __future__ import annotations

from datasets import load_dataset


def load_prompts(dataset: str, subset: str | None, split: str, num_prompts: int,
                 min_chars: int = 200) -> list[str]:
    """Primeros ``num_prompts`` textos con al menos ``min_chars`` caracteres."""
    ds = load_dataset(dataset, subset, split=split)
    prompts = [t for t in ds["text"] if len(t.strip()) >= min_chars]
    return prompts[:num_prompts]
