"""Region zoom: crop-and-rescale selected ROIs (strategy 1a).

Given ROI boundaries derived from the LGRE existence score map, crop the ROI
from the original image, upscale it (``zoom_scale``) and run detection. Boxes
produced in the zoomed frame must be mapped back to the original image frame
before global NMS.

This is a standard "slicing / zoom-in" strategy (cf. SAHI, Akyon et al. 2022);
the causal contribution is *which* ROIs are chosen.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RegionZoom(nn.Module):
    """Crop and rescale regions of interest for small-object detection."""

    def __init__(self, zoom_scale: float = 2.0, min_roi_pixels: int = 16) -> None:
        """Initialise the region zoom module.

        Args:
            zoom_scale: Upscaling factor applied to each cropped ROI.
            min_roi_pixels: Minimum ROI side length; smaller ROIs are skipped.
        """
        super().__init__()
        self.zoom_scale = zoom_scale
        self.min_roi_pixels = min_roi_pixels

    def crop_and_rescale(self, image: torch.Tensor, rois: torch.Tensor) -> list[torch.Tensor]:
        """Crop each ROI and rescale it by ``zoom_scale``.

        Args:
            image: Source image, shape ``[3, H, W]``.
            rois: ROI boxes in ``xyxy`` format, shape ``[R, 4]``.

        Returns:
            List of ``R`` rescaled image patches.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("RegionZoom.crop_and_rescale is not implemented yet")

    def map_back(
        self,
        boxes_zoomed: torch.Tensor,
        roi: torch.Tensor,
    ) -> torch.Tensor:
        """Map boxes from the zoomed frame back to the original frame.

        Args:
            boxes_zoomed: Boxes in ``xyxy`` in the zoomed ROI frame,
                shape ``[M, 4]``.
            roi: The ROI (``xyxy``) used for cropping, shape ``[4]``.

        Returns:
            Boxes in the original image frame, shape ``[M, 4]``.

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError("RegionZoom.map_back is not implemented yet")
