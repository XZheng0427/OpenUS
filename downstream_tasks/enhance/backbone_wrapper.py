"""Re-export VSSMFeatureExtractor from the landmark pipeline.

The image-enhancement task consumes the same 4-stage VSSM feature pyramid
as the landmark task does, so we share the wrapper rather than duplicate it.
"""

from downstream_tasks.landmark.backbone_wrapper import VSSMFeatureExtractor

__all__ = ["VSSMFeatureExtractor"]
