import math

import torch

from layercollapse.metrics import curvature, effective_rank, matrix_entropy


def test_rank_one_has_zero_entropy_and_unit_rank():
    u = torch.randn(32, 1, dtype=torch.float64)
    Z = u @ torch.randn(1, 64, dtype=torch.float64)          # todos los tokens en la misma dirección
    assert matrix_entropy(Z) < 1e-6
    assert abs(effective_rank(Z) - 1.0) < 1e-6


def test_orthogonal_tokens_have_max_entropy():
    Z = torch.eye(16, 64)               # espectro plano
    assert abs(matrix_entropy(Z) - 1.0) < 1e-6
    assert abs(effective_rank(Z) - 16.0) < 1e-6


def test_renyi_alpha2_flat_spectrum():
    Z = torch.eye(8, 32)
    assert abs(matrix_entropy(Z, alpha=2.0) - 1.0) < 1e-6


def test_straight_line_has_zero_curvature():
    t = torch.arange(10, dtype=torch.float64).unsqueeze(1)
    Z = t * torch.randn(1, 16, dtype=torch.float64)
    assert curvature(Z) < 1e-6


def test_zigzag_curvature():
    Z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    assert abs(curvature(Z) - math.pi / 2) < 1e-6
