"""Legacy AerosolSegmenter for backward compatibility."""
import cv2
import numpy as np

from a_em.core import AerosolObject


class AerosolSegmenter:
    """
    Extracts binary masks for aerosol particles from preprocessed images.
    """
    @staticmethod
    def extract_mask(img, rect=None, adaptive=False, refine=False):
        """
        Main interface for mask extraction.

        Args:
            img: Preprocessed grayscale image.
            rect: Manual Bounding Box (x, y, w, h).
            adaptive: If True, use adaptive thresholding.
            refine: If True, refine mask using automatic GrabCut.
        """
        if rect and len(rect) == 4:
            mask = AerosolSegmenter._grabcut(img, rect)
        else:
            mask = AerosolSegmenter._threshold_segment(img, adaptive=adaptive)

        if refine and not rect:
            mask = AerosolSegmenter.refine_mask(img, mask)

        return mask

    @staticmethod
    def _grabcut(img, rect):
        mask = np.zeros(img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
        cv2.grabCut(img_color, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return mask2 * 255

    @staticmethod
    def _threshold_segment(img, adaptive=False):
        if adaptive:
            mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 2)
        else:
            _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    @staticmethod
    def refine_mask(img, initial_mask):
        """
        Automatically refines a mask using GrabCut based on its contours.
        """
        refined_mask = np.zeros_like(initial_mask)
        contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Expand bbox slightly for GrabCut
            pad = 5
            rect = (max(0, x-pad), max(0, y-pad),
                    min(img.shape[1]-x, w+2*pad), min(img.shape[0]-y, h+2*pad))

            mask = np.zeros(img.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            try:
                cv2.grabCut(img_color, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                refined_mask = cv2.bitwise_or(refined_mask, mask2 * 255)
            except cv2.error:
                # Fallback to initial mask for this contour if GrabCut fails
                cv2.drawContours(refined_mask, [cnt], -1, 255, -1)

        return refined_mask

    @staticmethod
    def extract_all_objects(original_img, full_mask, min_area=100):
        """
        Analyzes connected components, filters noise, and returns a list of AerosolObject.
        """
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        aerosols = []

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Extract ROI
            img_roi = original_img[y:y+h, x:x+w]
            mask_roi = full_mask[y:y+h, x:x+w].copy()

            # Clean mask ROI to only contain the current contour
            local_cnt = cnt - [x, y]
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(temp_mask, [local_cnt], -1, 255, -1)
            mask_roi = cv2.bitwise_and(mask_roi, temp_mask)

            obj = AerosolObject(
                aerosol_id=i+1,
                image_roi=img_roi,
                mask_roi=mask_roi,
                offset=(y, x),
                bbox=(x, y, w, h),
                area=area
            )
            aerosols.append(obj)

        return aerosols
