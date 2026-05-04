"""Abstract base class for segmentation backends."""
from abc import ABC, abstractmethod
import numpy as np

from atem_analyzer.core import AerosolObject
from atem_analyzer.config import PipelineConfig


class SegmentationBackend(ABC):
    name: str = 'abstract'

    @abstractmethod
    def segment(self, signal, config: PipelineConfig) -> np.ndarray:
        pass

    @classmethod
    def supports(cls, microscope_type: str, particle_type: str) -> bool:
        return True

    def extract_objects(self, original_signal, mask: np.ndarray,
                        config: PipelineConfig) -> list:
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
                import hyperspy.api as hs
                roi_signal = original_signal.isig[x:x + bw, y:y + bh]
            except Exception:
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
