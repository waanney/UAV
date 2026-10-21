"""Confounder dictionary ``Z`` and its prior ``P(Z)`` (MCR-UOD Sec 4.2).

The dictionary is built from CLIP text embeddings of condition descriptions,
because collecting UAV images for every shooting condition is infeasible::

    Z in R^{S x D2},  S = num_classes * K * M

where ``K`` is the number of weather classes and ``M`` the number of
(occlusion x view) combinations (see ``configs/mcr_causal.yaml`` and
``confunder.txt``).

Definitions referenced below are from the paper:
    Eq.(1)  P(Y|do(F)) = sum_z P(Y|F,Z=z) P(Z=z)
    Eq.(11) z_i^{t+1} = alpha * z_i^t + (1 - alpha) * F_p
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConfounderDictionary(nn.Module):
    """Learnable confounder dictionary initialised from CLIP text prompts.

    Attributes:
        num_classes: Number of detector classes (``nc``).
        d2: Unified text/image embedding dimension (``D2``).
        K: Number of weather strata.
        M: Number of (occlusion x view) strata.
        S: Dictionary size, ``num_classes * K * M``.
        Z: Text-embedding bank, lazily built with shape ``[S, D2]``.
        prior_logits: Learnable prior parameters, shape ``[S]`` (if
            ``learnable_prior``), else ``None``.
    """

    def __init__(
        self,
        num_classes: int,
        d2: int,
        descriptions_path: str,
        K: int = 4,
        M: int = 9,
        learnable_prior: bool = True,
    ) -> None:
        """Initialise the dictionary.

        Args:
            num_classes: Number of detector classes.
            d2: Unified embedding dimension.
            descriptions_path: Path to the confounder prompt bank
                (``confunder.txt``).
            K: Number of weather strata.
            M: Number of (occlusion x view) strata.
            learnable_prior: If True, ``P(Z)`` is a learnable softmax over
                ``prior_logits``; otherwise the prior is set empirically.
        """
        super().__init__()
        self.num_classes = num_classes
        self.d2 = d2
        self.K = K
        self.M = M
        self.S = num_classes * K * M
        self.descriptions_path = descriptions_path

        self.register_buffer("Z", torch.empty(0), persistent=False)
        if learnable_prior:
            self.prior_logits: nn.Parameter | None = nn.Parameter(torch.zeros(self.S))
        else:
            self.prior_logits = None

    def build_text_bank(self, clip_text_encoder=None) -> torch.Tensor:
        """Build ``Z`` from ``descriptions_path`` using a CLIP text encoder.

        Steps (TODO):
            1. Read the K*M condition descriptions from ``descriptions_path``.
            2. For every class and every description, form
               ``f"{cls}, {description}"`` -> ``S`` prompts.
            3. Encode + L2-normalise with the CLIP text encoder.
            4. Store as ``self.Z`` with shape ``[S, D2]``.

        Args:
            clip_text_encoder: Callable mapping ``list[str] -> Tensor[S, D2]``.

        Returns:
            The dictionary ``Z`` of shape ``[S, D2]``.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("ConfounderDictionary.build_text_bank is not implemented yet")

    def prior(self) -> torch.Tensor:
        """Return the prior distribution ``P(Z)`` over dictionary items.

        Returns:
            ``P(Z)`` of shape ``[S]`` summing to 1.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("ConfounderDictionary.prior is not implemented yet")

    def update(self, F_p: torch.Tensor, z_idx: torch.Tensor) -> None:
        """Update dictionary items with image features (Eq.11, in place).

        ``z_i^{t+1} = alpha * z_i^t + (1 - alpha) * F_p``

        Args:
            F_p: Selected visual features, shape ``[N, D2]``.
            z_idx: Target dictionary index per feature, shape ``[N]``.
                With ``argmax`` assignment, ``z_idx`` is the nearest entry per
                feature; with ``threshold`` assignment, features below the
                similarity threshold are skipped.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("ConfounderDictionary.update is not implemented yet")

    def nearest(self, F: torch.Tensor) -> torch.Tensor:
        """Return the index of the most similar dictionary item per feature.

        Args:
            F: Feature matrix, shape ``[N, D2]``.

        Returns:
            ``argmax_z cos(F, Z)`` of shape ``[N]``.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("ConfounderDictionary.nearest is not implemented yet")
