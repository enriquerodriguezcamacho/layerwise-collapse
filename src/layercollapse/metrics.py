"""Métricas de calidad de representación por capa.

Convención: ``Z`` es la matriz de representaciones de un prompt en una capa,
de forma (N, D): N tokens, D dimensión oculta.
"""

from __future__ import annotations

import torch


def _normalized_gram_eigvals(Z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Autovalores de K = Z Z^T / tr(Z Z^T), que suman 1.

    Los autovalores no nulos de Z Z^T coinciden con los de Z^T Z, así que se usa
    la matriz más pequeña de las dos.
    """
    Z = Z.double()
    G = Z @ Z.T if Z.shape[0] <= Z.shape[1] else Z.T @ Z
    lam = torch.linalg.eigvalsh(G).clamp_min(0.0)
    return lam / lam.sum().clamp_min(eps)


def matrix_entropy(Z: torch.Tensor, alpha: float = 1.0, normalize: bool = True) -> float:
    """Entropía matricial de Rényi de orden alpha sobre la Gram normalizada.

    S_alpha(K) = 1/(1-alpha) * log sum_i lambda_i^alpha   (alpha != 1)
    S_1(K)     = -sum_i lambda_i log lambda_i              (límite alpha -> 1)

    Con ``normalize=True`` se divide por log(rango máximo) para obtener un valor en [0, 1].
    Aplicada a los tokens de un prompt es la "prompt entropy" del paper; aplicada a las
    representaciones promedio de muchos prompts es la "dataset entropy".
    """
    lam = _normalized_gram_eigvals(Z)
    lam = lam[lam > 0]
    if abs(alpha - 1.0) < 1e-8:
        H = -(lam * lam.log()).sum()
    else:
        H = torch.log((lam**alpha).sum()) / (1.0 - alpha)
    if normalize:
        r = min(Z.shape)
        H = H / torch.log(torch.tensor(float(r), dtype=H.dtype)) if r > 1 else H * 0
    return H.item()


def effective_rank(Z: torch.Tensor, center: bool = False) -> float:
    """Rango efectivo (Roy & Vetterli, 2007): exp(entropía de Shannon de los
    valores singulares normalizados). Vale 1 si hay colapso total y min(N, D) si el
    espectro es plano.
    """
    Z = Z.double()
    if center:
        Z = Z - Z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Z)
    p = s / s.sum().clamp_min(1e-12)
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum()).item()


def curvature(Z: torch.Tensor, eps: float = 1e-12) -> float:
    """Curvatura media de la trayectoria de tokens (en radianes).

    v_k = z_{k+1} - z_k ;  c_k = arccos( <v_k, v_{k+1}> / (||v_k|| ||v_{k+1}||) )
    Devuelve la media de c_k. Requiere N >= 3.
    """
    if Z.shape[0] < 3:
        raise ValueError("curvature necesita al menos 3 tokens")
    V = (Z[1:] - Z[:-1]).double()
    num = (V[:-1] * V[1:]).sum(dim=1)
    den = V[:-1].norm(dim=1) * V[1:].norm(dim=1)
    cos = (num / den.clamp_min(eps)).clamp(-1.0, 1.0)
    return torch.arccos(cos).mean().item()


# TODO: métricas basadas en augmentations del paper (InfoNCE, LiDAR, DiME).
