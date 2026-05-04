"""Segmentation backends for aerosol particle extraction."""
from atem_analyzer.segmentation.base import SegmentationBackend
from atem_analyzer.segmentation.registry import SegmentationRegistry
from atem_analyzer.segmentation.traditional import TraditionalCVSegmenter
from atem_analyzer.segmentation.legacy import AerosolSegmenter

SegmentationRegistry.register(TraditionalCVSegmenter)

__all__ = [
    'SegmentationBackend',
    'SegmentationRegistry',
    'TraditionalCVSegmenter',
    'AerosolSegmenter',
]
