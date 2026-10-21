"""Language-Guided Region Exploration (LGRE), MCR-UOD Sec 4.1 / Eq.(5).

LGRE uses CLIP text embeddings to score how likely each spatial location is to
contain an object, producing an *object existence map* ``s``. The top-``tau``
fraction of pixels become the feature matrix ``F`` fed to BACR::

    F = { C'_1(i) | i in I_tau },  I_tau = TopN(s),  N = tau * H * W    (Eq.5)

NOTE: in the upstream code this logic is split across ``MaxSigmoidAttnBlock``
(which returns the attention score map) and ``select_top_k_pixels`` in
``ultralytics/nn/modules/block.py``. Here it is reimplemented cleanly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LGRE(nn.Module):
    """Compute the object existence score map ``s`` from visual + text features."""

    def __init__(self, d2: int = 512, num_heads: int = 1) -> None:
        """Initialise LGRE.

        Args:
            d2: Unified embedding dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.d2 = d2
        self.num_heads = num_heads

    def forward(self, visual_feat: torch.Tensor, txt_feats: torch.Tensor) -> torch.Tensor:
        """Compute the object existence score map.

        Args:
            visual_feat: Feature map, shape ``[B, C, H, W]``.
            txt_feats: Class text embeddings, shape ``[1, nc, D2]``.

        Returns:
            Existence score map ``s`` in ``[0, 1]``, shape ``[B, 1, H, W]``.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("LGRE.forward is not implemented yet")


def topk_select(score: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the top-``tau`` fraction of pixels by existence score (Eq.5).

    Args:
        score: Existence score map, shape ``[B, 1, H, W]``.
        tau: Selection ratio in ``(0, 1]``; ``N = int(tau * H * W)``.

    Returns:
        A tuple ``(F, I)`` where:

            - ``F``: selected features, shape ``[B, N, D2]``.
            - ``I``: flat spatial indices for scatter-back, shape ``[B, N]``.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError("topk_select is not implemented yet")
