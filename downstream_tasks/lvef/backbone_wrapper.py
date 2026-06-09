"""Re-export the VMamba feature extractor from the landmark module.

The LVEF pipeline uses the exact same feature contract as the landmark
pipeline (4-stage spatial pyramid from ``Backbone_DINOv2_VSSM_2``), so we
share the implementation rather than copy it.
"""

from downstream_tasks.landmark.backbone_wrapper import VSSMFeatureExtractor

__all__ = ["VSSMFeatureExtractor"]
