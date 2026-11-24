"""Carga de modelos y extracción de hidden states por capa."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(name: str, revision: str = "main", dtype: str = "float32", device: str = "auto"):
    """Carga modelo y tokenizer de Hugging Face. Funciona igual para Pythia, Llama o Mamba (-hf)."""
    dev = resolve_device(device)
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=getattr(torch, dtype)
    ).to(dev)
    model.eval()
    return model, tok, dev


@torch.no_grad()
def hidden_states(model, tok, text: str, device, max_length: int = 128) -> list[torch.Tensor]:
    """Devuelve una lista de L+1 tensores (N, D): embeddings + salida de cada capa."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out = model(**enc, output_hidden_states=True)
    return [h[0].float().cpu() for h in out.hidden_states]
