"""Feature zoom: causal magnification of small-object features (strategy 1b).

Instead of cropping pixels, this module enlarges the *representation* of
distant/small objects so their features resemble those of near/large objects.
The scale / distance factor is treated as a confounder and removed with the
backdoor adjustment, reusing the shared confounder dictionary and BACR::

    F'_small = E_z[ f(F_small, z) ],   z ~ P(Z_scale)

The corrected features are then written back to the feature map at the selected
indices, mirroring the BACR scatter-back.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureZoom(nn.Module):
    """Magnify small-object features using causal de-biasing over scale."""

    def __init__(self, d2: int = 512, num_heads: int = 1) -> None:
        """Initialise the feature zoom module.

        Args:
            d2: Unified embedding dimension.
            num_heads: Attention heads used for the causal aggregation.
        """
        super().__init__()
        self.d2 = d2
        self.num_heads = num_heads

    def forward(
        self,
        feat: torch.Tensor,
        roi_mask: torch.Tensor,
        confounder: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the scale-de-biased feature map.

        Args:
            feat: Feature map, shape ``[B, C, H, W]``.
            roi_mask: Binary mask of small-object regions, shape ``[B, 1, H, W]``.
            confounder: Scale-confounder dictionary, shape ``[S, D2]``.
            prior: Prior ``P(Z_scale)``, shape ``[S]``.

        Returns:
            Feature map with magnified small-object features,
            shape ``[B, C, H, W]``.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("FeatureZoom.forward is not implemented yet")
