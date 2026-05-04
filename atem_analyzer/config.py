"""Configuration classes for the ATEM analysis pipeline."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PipelineConfig:
    """Central configuration that drives the entire analysis pipeline."""

    microscope_type: str = 'TEM'
    particle_type: str = 'soot'

    # Preprocessing
    clahe_clip: float = 3.0
    clahe_tile: Tuple[int, int] = (8, 8)
    background_kernel: int = 25
    filter_type: str = 'bilateral'

    # Segmentation
    segmentation_backend: str = 'traditional_cv'
    threshold_method: str = 'adaptive'
    min_area: int = 100
    use_grabcut_refinement: bool = True

    # Analysis
    analysis_engine: str = 'soot'
    dist_thresh_ratio: float = 0.4

    # Reporting
    output_dir: str = 'data/processed'

    @classmethod
    def from_signal(cls, signal, particle_type: str = 'soot'):
        md = signal.metadata
        mtype = 'TEM'
        if 'Acquisition_instrument' in md:
            inst = md.Acquisition_instrument
            if 'SEM' in inst:
                mtype = 'SEM'
            elif 'TEM' in inst:
                mtype = 'TEM'
        return cls(microscope_type=mtype, particle_type=particle_type)
