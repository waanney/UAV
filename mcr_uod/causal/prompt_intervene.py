"""Causal intervention over the prompt bank (research objective MT2).

Goal: distinguish visually similar classes (e.g. car vs truck, bus vs truck) by
treating *prompt selection* as the treatment ``T`` and adjusting for the
shooting-condition confounder ``Z`` with the backdoor formula::

    P(Y|do(T)) = sum_z P(Y | T, Z=z) P(Z=z)

This keeps the "prompt engineering" nature of MT2 while giving it a causal
grounding: instead of a single fixed prompt, classification logits are averaged
over confounder contexts weighted by ``P(Z)``.

Related reference: CoOp / "Learning to Prompt for Vision-Language Models"
(Zhou et al., IJCV 2022) for the prompt-tuning baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PromptBank(nn.Module):
    """Attribute-enriched prompt bank to separate visually similar classes."""

    def __init__(self, class_names: list[str], template: str, attributes: list[str]) -> None:
        """Build the prompt bank.

        Args:
            class_names: Detector class names.
            template: Prompt template with ``{cls}`` and ``{attr}`` slots,
                e.g. ``"a photo of a {cls}, {attr}"``.
            attributes: Attribute phrases used to disambiguate similar classes
                (colour, shape, scale, typical view, ...).
        """
        super().__init__()
        self.class_names = class_names
        self.template = template
        self.attributes = attributes

    def prompts(self) -> list[str]:
        """Return the flattened prompt list for CLIP text encoding.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("PromptBank.prompts is not implemented yet")


class PromptIntervener(nn.Module):
    """Apply the do-operator over prompts using the confounder prior."""

    def forward(
        self,
        logits: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        """Return the intervended class logits.

        Args:
            logits: Per-context classification logits, shape ``[B, S, nc]``
                (``S`` confounder strata).
            prior: Prior distribution ``P(Z)``, shape ``[S]``.

        Returns:
            Logits after marginalising over ``Z``, shape ``[B, nc]``.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("PromptIntervener.forward is not implemented yet")
