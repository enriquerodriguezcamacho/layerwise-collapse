"""Objetivo (ii): normas de gradiente en las proyecciones query/key por capa.

No requiere entrenar: basta un forward + backward sobre un batch con la loss de
language modeling y leer ``param.grad`` de W_Q y W_K en cada bloque de atención.

Notas por arquitectura:
- Pythia (GPTNeoX): Q, K y V van fusionadas en ``attention.query_key_value``; hay que
  separar el gradiente por bloques de filas.
- Llama: ``self_attn.q_proj`` y ``self_attn.k_proj`` por separado.
- Mamba: no hay Q/K; el análogo serían las proyecciones B y C del SSM (``x_proj``).
"""

from __future__ import annotations


def qk_grad_norms(model, batch) -> dict[int, dict[str, float]]:
    """Devuelve {capa: {"q": ||dL/dW_Q||, "k": ||dL/dW_K||}}."""
    raise NotImplementedError
