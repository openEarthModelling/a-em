"""Analysis engines for aerosol particle characterization."""
from atem_analyzer.analysis.base import AnalysisEngine, AerosolAnalysisEngine
from atem_analyzer.analysis.registry import AnalysisEngineRegistry
from atem_analyzer.analysis.soot import SootAnalysisEngine

# Auto-register built-in engines
AnalysisEngineRegistry.register(SootAnalysisEngine)

__all__ = [
    'AnalysisEngine',
    'AerosolAnalysisEngine',
    'AnalysisEngineRegistry',
    'SootAnalysisEngine',
]
