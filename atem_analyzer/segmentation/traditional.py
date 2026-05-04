"""Traditional CV-based segmentation for soot aggregates."""
import cv2
import numpy as np

from atem_analyzer.segmentation.base import SegmentationBackend
from atem_analyzer.io import HyperSpyReader
from atem_analyzer.config import PipelineConfig


class TraditionalCVSegmenter(SegmentationBackend):
    name = 'traditional_cv'

    @classmethod
    def supports(cls, microscope_type: str, particle_type: str) -> bool:
        return particle_type == 'soot'

    def segment(self, signal, config: PipelineConfig) -> np.ndarray:
        s8 = HyperSpyReader.to_uint8(signal)
        img = s8.data

        enhanced = self._apply_clahe(img, config.clahe_clip, config.clahe_tile)
        bg_removed = self._remove_background(enhanced, config.background_kernel)

        if config.filter_type == 'bilateral':
            denoised = cv2.bilateralFilter(bg_removed, 9, 75, 75)
        else:
            denoised = cv2.GaussianBlur(bg_removed, (3, 3), 0)

        if config.threshold_method == 'adaptive':
            mask = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 21, 2)
        else:
            _, mask = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if config.use_grabcut_refinement:
            mask = self._refine_with_grabcut(denoised, mask)

        return mask

    @staticmethod
    def _apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

    @staticmethod
    def _remove_background(img, kernel_size=25):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    @staticmethod
    def _refine_with_grabcut(img, initial_mask):
        refined_mask = np.zeros_like(initial_mask)
        contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            pad = 5
            rect = (max(0, x - pad), max(0, y - pad),
                    min(img.shape[1] - x, w + 2 * pad),
                    min(img.shape[0] - y, h + 2 * pad))
            mask_gc = np.zeros(img.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(img_color, mask_gc, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype('uint8')
                refined_mask = cv2.bitwise_or(refined_mask, mask2 * 255)
            except cv2.error:
                cv2.drawContours(refined_mask, [cnt], -1, 255, -1)

        return refined_mask
