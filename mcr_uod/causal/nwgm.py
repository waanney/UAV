"""Normalised Weighted Geometric Mean (NWGM) approximation, MCR-UOD Eq.(7).

The causal intervention marginalises over confounders inside a softmax::

    E_z[Softmax(f(F, z))]  ~=NWGM~  Softmax(E_z[f(F, z)])            (Eq.7)

NWGM moves the outer expectation inside the softmax so the intervention can be
computed with a single forward pass. This is the piece missing from the upstream
implementation (which only fuses features once instead of marginalising).
"""

from __future__ import annotations

from typing import Callable

import torch


def nwgm_intervene(
    F: torch.Tensor,
    Z: torch.Tensor,
    prior: torch.Tensor,
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Estimate ``E_z[f(F, z)]`` for the NWGM intervention (Eq.7).

    Args:
        F: Selected features, shape ``[B, N, D2]``.
        Z: Confounder dictionary, shape ``[S, D2]``.
        prior: Prior distribution ``P(Z)``, shape ``[S]``.
        f: Pairwise scoring function ``f(F, z) -> logits`` used inside the
            softmax of Eq.(6).

    Returns:
        The prior-weighted expectation ``E_z[f(F, z)]`` of the same shape as
        ``f`` output.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("nwgm_intervene is not implemented yet")
