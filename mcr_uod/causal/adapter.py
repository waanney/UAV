"""Adapter connecting the causal half with the uncertainty half.

The uncertainty team owns ``mcr_uod.uncertainty`` (Conformal Prediction +
Rerank). This adapter is the single join point between the two halves and is
created by the causal team once the uncertainty interface is frozen.

Pipeline (research proposal, stages 3-4)::

    image -> [causal: zoom + prompt] -> detector -> raw boxes/scores
          -> [uncertainty: conformal calibrate + rerank] -> final detections

Nothing is wired yet; the interface is intentionally "TBD" until both teams
agree on the detection format.
"""

from __future__ import annotations

from typing import Any


class CausalUncertaintyAdapter:
    """Join causal detector outputs with the uncertainty calibration stage."""

    def __init__(self, causal_model: Any, calibrator: Any = None, reranker: Any = None) -> None:
        """Initialise the adapter.

        Args:
            causal_model: The causal-enhanced detector (YOLOWorld + causal
                modules).
            calibrator: Conformal calibrator from ``mcr_uod.uncertainty``
                (``None`` until that module exists).
            reranker: Reranker from ``mcr_uod.uncertainty`` (``None`` until that
                module exists).
        """
        self.causal_model = causal_model
        self.calibrator = calibrator
        self.reranker = reranker

    def predict(self, images: Any) -> Any:
        """Run the causal detector and calibrate + rerank its detections.

        Args:
            images: Input batch (paths or tensors) accepted by the detector.

        Returns:
            Final detections with calibrated confidences.

        Raises:
            NotImplementedError: Always, until both halves are integrated.
        """
        raise NotImplementedError("CausalUncertaintyAdapter.predict is not implemented yet")
