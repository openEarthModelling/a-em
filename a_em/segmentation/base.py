"""Abstract base class for segmentation backends."""
from abc import ABC, abstractmethod
import numpy as np

from a_em.core import AerosolObject
from a_em.config import PipelineConfig


class SegmentationBackend(ABC):
    """Abstract base class for all segmentation algorithms."""

    name: str = 'abstract'

    @abstractmethod
    def segment(self, signal, config: PipelineConfig) -> np.ndarray:
        """Segment signal and return a binary mask (0/255).

        Args:
            signal: HyperSpy Signal2D or numpy array containing the image.
            config: Pipeline configuration with segmentation parameters.

        Returns:
            Binary mask as uint8 array with values 0 or 255.
        """
        pass

    @classmethod
    def supports(cls, microscope_type: str, particle_type: str) -> bool:
        """Check if this backend supports the given microscope/particle combination.

        Args:
            microscope_type: Type of microscope (e.g., 'TEM', 'SEM').
            particle_type: Type of particle (e.g., 'soot', 'spherical').

        Returns:
            False by default. Subclasses must override to opt-in.
        """
        return False

    def extract_objects(self, original_signal, mask: np.ndarray,
                        config: PipelineConfig) -> list[AerosolObject]:
        """Extract individual aerosol objects from a binary mask.

        Performs connected component analysis and returns a list of
        AerosolObject instances, one per detected contour.

        Args:
            original_signal: The original image signal (HyperSpy or numpy).
            mask: Binary mask with detected regions (0/255).
            config: Pipeline configuration with min_area threshold.

        Returns:
            List of AerosolObject instances, sorted by contour order.
        """
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        aerosols = []
        h, w = mask.shape

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < config.min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            x = max(0, x)
            y = max(0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            roi_data = original_signal.data[y:y + bh, x:x + bw]
            mask_roi = mask[y:y + bh, x:x + bw].copy()

            local_cnt = cnt - [x, y]
            temp_mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(temp_mask, [local_cnt], -1, 255, -1)
            mask_roi = cv2.bitwise_and(mask_roi, temp_mask)

            try:
                roi_signal = original_signal.isig[x:x + bw, y:y + bh]
            except (ImportError, AttributeError):
                roi_signal = roi_data

            obj = AerosolObject(
                aerosol_id=i + 1,
                image_roi=roi_signal,
                mask_roi=mask_roi,
                offset=(y, x),
                bbox=(x, y, bw, bh),
                area=area
            )
            aerosols.append(obj)

        return aerosols
