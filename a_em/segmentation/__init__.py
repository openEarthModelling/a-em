"""Segmentation backends for aerosol particle extraction."""
from a_em.segmentation.base import SegmentationBackend
from a_em.segmentation.registry import SegmentationRegistry
from a_em.segmentation.traditional import TraditionalCVSegmenter
from a_em.segmentation.legacy import AerosolSegmenter
from a_em.segmentation.sam_automask import SAMAutoMaskSegmenter

SegmentationRegistry.register(TraditionalCVSegmenter)
SegmentationRegistry.register(SAMAutoMaskSegmenter)

__all__ = [
    'SegmentationBackend',
    'SegmentationRegistry',
    'TraditionalCVSegmenter',
    'AerosolSegmenter',
    'SAMAutoMaskSegmenter',
]
