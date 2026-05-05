"""Segmentation backends for aerosol particle extraction."""
from atem_analyzer.segmentation.base import SegmentationBackend
from atem_analyzer.segmentation.registry import SegmentationRegistry
from atem_analyzer.segmentation.traditional import TraditionalCVSegmenter
from atem_analyzer.segmentation.legacy import AerosolSegmenter
from atem_analyzer.segmentation.sam_automask import SAMAutoMaskSegmenter

SegmentationRegistry.register(TraditionalCVSegmenter)
SegmentationRegistry.register(SAMAutoMaskSegmenter)

__all__ = [
    'SegmentationBackend',
    'SegmentationRegistry',
    'TraditionalCVSegmenter',
    'AerosolSegmenter',
    'SAMAutoMaskSegmenter',
]
