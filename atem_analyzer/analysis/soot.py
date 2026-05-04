"""Soot aggregate analysis: watershed segmentation and fractal dimension."""
import cv2
import numpy as np
from scipy.stats import linregress

from atem_analyzer.analysis.base import AnalysisEngine


class SootAnalysisEngine(AnalysisEngine):
    """Engine for analyzing soot aggregates in TEM/SEM images.

    Performs watershed segmentation to split aggregates into primary particles,
    then calculates geometric metrics and fractal dimension (Df).
    """

    name = 'soot'

    def __init__(self, dist_thresh_ratio=0.4):
        self.dist_thresh_ratio = dist_thresh_ratio

    @classmethod
    def supports(cls, particle_type: str) -> bool:
        return particle_type == 'soot'

    def analyze(self, signal_roi, mask_roi) -> dict:
        """Complete soot analysis pipeline.

        Args:
            signal_roi: HyperSpy Signal2D or numpy array of the particle ROI.
            mask_roi: Binary mask (uint8) of the particle.

        Returns:
            Dictionary with keys: df, rg, r0, feret_dia, convexity,
            roundness, num_particles, primary_particles.
        """
        # Extract numpy data for OpenCV operations
        if isinstance(signal_roi, np.ndarray):
            img = signal_roi
        elif hasattr(signal_roi, 'data'):
            img = np.asarray(signal_roi.data)
        else:
            img = np.asarray(signal_roi)

        if img.dtype != np.uint8:
            # Normalize to uint8 for OpenCV
            dmin, dmax = img.min(), img.max()
            if dmax == dmin:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = ((img - dmin) / (dmax - dmin) * 255).astype(np.uint8)

        # 1. Watershed segmentation of primary particles
        markers, dist_transform = self._split_particles(mask_roi)
        particles = self._extract_particle_stats(markers, dist_transform)

        # 2. Calculate all metrics
        metrics = self._calculate_all_metrics(particles, mask_roi)
        if not metrics:
            return {}
        metrics['primary_particles'] = particles
        metrics['markers'] = markers
        metrics['dist_transform'] = dist_transform

        return metrics

    def _split_particles(self, mask):
        """Watershed segmentation to split aggregate into primary particles."""
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist_transform,
            self.dist_thresh_ratio * dist_transform.max(),
            255, 0
        )
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        return markers, dist_transform

    def _extract_particle_stats(self, markers, dist_transform):
        """Extract center and radius for each primary particle."""
        particles = []
        for label in np.unique(markers):
            if label <= 1:
                continue
            particle_mask = np.uint8(markers == label)
            M = cv2.moments(particle_mask)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            max_dist = np.max(dist_transform[markers == label])
            particles.append({
                "id": int(label),
                "center": (cX, cY),
                "radius": float(max_dist),
                "area": float(M["m00"])
            })
        return particles

    def _calculate_all_metrics(self, particles, mask):
        """Calculate geometric metrics and fractal dimension."""
        if not particles:
            return {}

        # Weighted center of aggregate
        avg_r = np.mean([p['radius'] for p in particles])
        sum_m, sum_x, sum_y = 0, 0, 0
        for p in particles:
            m = (p['radius'] / avg_r) ** 3
            sum_x += m * p['center'][0]
            sum_y += m * p['center'][1]
            sum_m += m
        center = (sum_x / sum_m, sum_y / sum_m)

        # Distance from center for each particle
        for p in particles:
            p['ri'] = np.sqrt(
                (p['center'][0] - center[0])**2 +
                (p['center'][1] - center[1])**2
            )
        particles.sort(key=lambda x: x['ri'])

        # Cumulative Rg and R0
        sum_r3_ri2, sum_r3, sum_r = 0, 0, 0
        rg_list, r0_list = [], []
        for i, p in enumerate(particles):
            m = (p['radius'] / avg_r) ** 3
            sum_r3_ri2 += m * (p['ri'] ** 2)
            sum_r3 += m
            sum_r += p['radius']
            rg_list.append(np.sqrt(sum_r3_ri2 / sum_r3))
            r0_list.append(sum_r / (i + 1))

        # Fractal dimension via log-log regression
        log_n = np.log10(np.arange(1, len(particles) + 1))
        log_rg_r0 = np.log10(np.array(rg_list) / np.array(r0_list))
        start_idx = max(5, int(len(particles) * 0.1))
        df_val = 0.0
        if len(log_n) > start_idx + 2:
            slope, _, _, _, _ = linregress(log_rg_r0[start_idx:], log_n[start_idx:])
            df_val = slope

        # Morphology
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(hull)
        rect = cv2.minAreaRect(cnt)
        feret_dia = max(rect[1])
        convexity = area / hull_area if hull_area > 0 else 0
        roundness = (4 * area) / (np.pi * (feret_dia ** 2)) if feret_dia > 0 else 0

        return {
            "center": center,
            "df": df_val,
            "rg": rg_list[-1],
            "r0": r0_list[-1],
            "feret_dia": feret_dia,
            "convexity": convexity,
            "roundness": roundness,
            "num_particles": len(particles),
            "log_data": (log_rg_r0, log_n)
        }
