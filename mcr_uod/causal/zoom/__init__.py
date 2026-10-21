"""Zoom modules for distant / small objects.

Two complementary strategies are scaffolded so both can be ablated:

    region_zoom  (1a) crop the ROI and rescale it before detection
    feature_zoom (1b) magnify small-object features in the feature space via
                      causal de-biasing over the scale confounder

The causal part is the *selection* of where to zoom (LGRE existence map ``s``);
the zoom itself is a region-based / feature-based operation.
"""

from __future__ import annotations

from mcr_uod.causal.zoom.feature_zoom import FeatureZoom
from mcr_uod.causal.zoom.region_zoom import RegionZoom

__all__ = ["RegionZoom", "FeatureZoom"]
