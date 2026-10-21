"""Backdoor Adjustment Causal Reasoning (BACR), MCR-UOD Sec 4.2 / Eq.(6-10).

Given selected features ``F`` (from LGRE) and the confounder dictionary ``Z``
with prior ``P(Z)``, BACR estimates the interventional distribution::

    P(Y|do(F)) = sum_z P(Y|F,Z=z) P(Z=z) = E_z[Softmax(f(F,z))]      (Eq.6)
               ~= Softmax(E_z[f(F,z)])                                (Eq.7, NWGM)

with cross-attention over the dictionary::

    Q = Wq F + bq,  K = Wk Z + bk,  V = Wv Z + bv                     (Eq.8)
    A = softmax(Q K^T / sqrt(D2))                                     (Eq.9)
    F' = f_F( Cat(F, A V^T) )                                         (Eq.10)

``F'`` is scattered back into the feature map at the LGRE indices ``I``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BACR(nn.Module):
    """Backdoor adjustment causal reasoning block."""

    def __init__(
        self,
        d2: int = 512,
        num_heads: int = 1,
        fuse_mode: str = "concat",
        use_nwgm: bool = True,
    ) -> None:
        """Initialise BACR.

        Args:
            d2: Unified embedding dimension (``D2``).
            num_heads: Number of attention heads for cross-attention.
            fuse_mode: ``"concat"`` implements Eq.(10) exactly; ``"residual"``
                reproduces the upstream approximation and is kept for ablation.
            use_nwgm: Whether to apply the NWGM approximation (Eq.7).
        """
        super().__init__()
        self.d2 = d2
        self.num_heads = num_heads
        self.fuse_mode = fuse_mode
        self.use_nwgm = use_nwgm
        self.query = nn.Linear(d2, d2)
        self.key = nn.Linear(d2, d2)
        self.value = nn.Linear(d2, d2)
        self.fuse = nn.Linear(2 * d2, d2) if fuse_mode == "concat" else None

    def forward(
        self,
        F: torch.Tensor,
        Z: torch.Tensor,
        prior: torch.Tensor,
        use_nwgm: bool | None = None,
    ) -> torch.Tensor:
        """Apply the do-operator to selected features.

        Args:
            F: Selected features, shape ``[B, N, D2]``.
            Z: Confounder dictionary, shape ``[S, D2]``.
            prior: Prior distribution ``P(Z)``, shape ``[S]``.
            use_nwgm: Overrides ``self.use_nwgm`` when provided.

        Returns:
            Intervened features ``F'`` of shape ``[B, N, D2]``.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("BACR.forward is not implemented yet")


def scatter_back(z: torch.Tensor, F_prime: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Interpolate ``F'`` back into the feature map at ``indices``.

    Args:
        z: Feature map to update, shape ``[B, C, H, W]``.
        F_prime: Intervened features, shape ``[B, N, C]``.
        indices: Flat spatial indices, shape ``[B, N]``.

    Returns:
        Updated feature map ``C''_1`` of shape ``[B, C, H, W]``.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("scatter_back is not implemented yet")
