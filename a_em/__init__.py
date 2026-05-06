"""ATEM Analyzer - Aerosol Electron Microscopy Image Analysis Platform."""
from .analysis import (
    AerosolAnalysisEngine,
    AnalysisEngine,
    AnalysisEngineRegistry,
    SootAnalysisEngine,
)
from .batch import BatchProcessor
from .config import PipelineConfig
from .core import AerosolObject
from .io import HyperSpyReader
from .pipeline import PipelineExecutor
from .preprocess import AerosolPreprocessor
from .reporter import AerosolReporter
from .segmentation import (
    AerosolSegmenter,
    SegmentationBackend,
    SegmentationRegistry,
    TraditionalCVSegmenter,
)

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
