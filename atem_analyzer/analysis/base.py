from abc import ABC, abstractmethod
import numpy as np

class AerosolAnalysisEngine(ABC):
    """
    Abstract base class for aerosol analysis algorithms.
    """
    @abstractmethod
    def analyze(self, image_roi: np.ndarray, mask_roi: np.ndarray) -> dict:
        """
        Analyzes a single aerosol particle's ROI and its mask.
        Returns a dictionary of calculated metrics.
        """
        pass
