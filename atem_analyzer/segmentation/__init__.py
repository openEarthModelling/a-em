"""Segmentation backends for aerosol particle extraction."""
from atem_analyzer.segmentation.base import SegmentationBackend
from atem_analyzer.segmentation.registry import SegmentationRegistry
from atem_analyzer.segmentation.traditional import TraditionalCVSegmenter

# Backward compatibility: legacy segmenter from old segmentation.py module
from atem_analyzer.segmentation_legacy import AerosolSegmenter

SegmentationRegistry.register(TraditionalCVSegmenter)

__all__ = [
    'SegmentationBackend',
    'SegmentationRegistry',
    'TraditionalCVSegmenter',
    'AerosolSegmenter',
]
