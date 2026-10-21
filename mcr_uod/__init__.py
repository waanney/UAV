"""MCR-UOD: Uncertainty-guided Causal Reasoning for Multimodal Object Detection.

Research package separated from the vendored ``ultralytics`` tree.

Ownership split (2 + 2):
    - ``mcr_uod.causal``       -> causal team (zoom + prompt de-confounding)
    - ``mcr_uod.uncertainty``  -> uncertainty team (conformal prediction + rerank)

This scaffold only defines interfaces and TODOs. Logic is implemented by each team.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
