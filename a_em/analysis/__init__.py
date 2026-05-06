"""Analysis engines for aerosol particle characterization."""
from a_em.analysis.base import AnalysisEngine, AerosolAnalysisEngine
from a_em.analysis.registry import AnalysisEngineRegistry
from a_em.analysis.soot import SootAnalysisEngine

# Auto-register built-in engines
AnalysisEngineRegistry.register(SootAnalysisEngine)

__all__ = [
    'AnalysisEngine',
    'AerosolAnalysisEngine',
    'AnalysisEngineRegistry',
    'SootAnalysisEngine',
]
