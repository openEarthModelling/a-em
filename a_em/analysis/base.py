"""Abstract base class for analysis engines."""
from abc import ABC, abstractmethod


class AnalysisEngine(ABC):
    """Abstract base class for aerosol particle analysis algorithms.

    Each engine analyzes a single AerosolObject and returns a dictionary
    of calculated metrics (e.g. fractal dimension, primary particle count).
    """

    name: str = 'abstract'

    @abstractmethod
    def analyze(self, signal_roi, mask_roi) -> dict:
        """Analyze a single aerosol particle.

        Args:
            signal_roi: HyperSpy Signal2D of the particle ROI, or numpy array.
            mask_roi: Binary mask (uint8) of the particle.

        Returns:
            Dictionary of calculated metrics.
        """
        pass

    @classmethod
    def supports(cls, particle_type: str) -> bool:
        """Check if this engine supports the given particle type.

        Override in subclasses to restrict applicability.

        Args:
            particle_type: e.g. 'soot', 'spherical'.

        Returns:
            False by default. Subclasses must override to opt-in.
        """
        return False


# Backward compatibility alias
AerosolAnalysisEngine = AnalysisEngine
