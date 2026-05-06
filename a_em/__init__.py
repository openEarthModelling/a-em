"""ATEM Analyzer - Aerosol Electron Microscopy Image Analysis Platform."""
from .io import HyperSpyReader
from .preprocess import AerosolPreprocessor
from .segmentation import (
    SegmentationBackend,
    SegmentationRegistry,
    TraditionalCVSegmenter,
    AerosolSegmenter,
)
from .core import AerosolObject
from .reporter import AerosolReporter
from .analysis import (
    AnalysisEngine,
    AerosolAnalysisEngine,
    AnalysisEngineRegistry,
    SootAnalysisEngine,
)
from .config import PipelineConfig
from .pipeline import PipelineExecutor
from .batch import BatchProcessor

__version__ = '0.1.0'

__all__ = [
    'PipelineConfig',
    'AerosolObject',
    'HyperSpyReader',
    'PipelineExecutor',
    'BatchProcessor',
    'AerosolReporter',
    'AerosolPreprocessor',
    'AerosolSegmenter',
    'SegmentationBackend',
    'SegmentationRegistry',
    'TraditionalCVSegmenter',
    'AnalysisEngine',
    'AerosolAnalysisEngine',
    'AnalysisEngineRegistry',
    'SootAnalysisEngine',
]
