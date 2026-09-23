"""Differentiable PyTorch SUPR foot fitting for prepared shoes.

A parallel implementation to ``foot_prior``'s NumPy search fitter, which is left
untouched and reused as the independent exact evaluator.

Pipeline:

    SUPR(betas, ankle/midfoot pitch, translation)   supr_torch.TorchSuprFoot
        -> foot vertices on the GPU
        -> differentiable containment / support losses     losses
           against a baked shoe cavity field                shoe_field
        -> autograd -> staged Adam                          fitter
        -> exact NumPy judge                                evaluate
"""

from .config import FitConfig, StageConfig
from .foot_fitter import FitResult, ShoeFootFitter
from .shoe_data import ShoeCase, available_shoes, load_shoe_case
from .cavity_field import (
    ShoeCavityField,
    build_cavity_field,
    sample_field,
    sample_open_above,
)
from .supr_torch import TorchSuprFoot

__all__ = [
    "FitConfig",
    "FitResult",
    "ShoeCase",
    "ShoeCavityField",
    "ShoeFootFitter",
    "StageConfig",
    "TorchSuprFoot",
    "available_shoes",
    "build_cavity_field",
    "load_shoe_case",
    "sample_field",
    "sample_open_above",
]
