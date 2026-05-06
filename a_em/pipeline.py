"""Single-image analysis pipeline executor."""
import cv2
import numpy as np

from a_em.analysis.registry import AnalysisEngineRegistry
from a_em.config import PipelineConfig
from a_em.io import HyperSpyReader
from a_em.reporter import AerosolReporter
from a_em.segmentation.registry import SegmentationRegistry


class PipelineExecutor:
    """Executes the full analysis pipeline on a single image.

    Pipeline stages:
        1. Preprocess (CLAHE, background removal, denoising)
        2. Segment -> binary mask
        3. Extract aerosol objects
        4. Analyze each object
        5. Report results
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.segmenter = SegmentationRegistry.get(config.segmentation_backend)
        self.engine = AnalysisEngineRegistry.get(config.analysis_engine)
        self.reporter = AerosolReporter(output_dir=config.output_dir)

    def run(self, signal):
        """Execute the full pipeline on a single Signal2D.

        Args:
            signal: HyperSpy Signal2D with calibrated axes.

        Returns:
            List of AerosolObject with populated metrics.
        """
        base_name = signal.metadata.General.title

        # 1. Preprocessing
        preprocessed = self._preprocess(signal)

        # 2. Segmentation
        mask = self.segmenter.segment(preprocessed, self.config)

        # 3. Extract objects
        aerosols = self.segmenter.extract_objects(signal, mask, self.config)

        # 4. Analyze
        for obj in aerosols:
            metrics = self.engine.analyze(obj.image_roi, obj.mask_roi)
            obj.metrics = metrics

        # 5. Report
        self._report_results(base_name, aerosols, signal)

        return aerosols

    def _preprocess(self, signal):
        """Apply preprocessing pipeline while preserving HyperSpy metadata.

        Args:
            signal: HyperSpy Signal2D.

        Returns:
            New Signal2D with preprocessed data.
        """
        # Convert to uint8 for OpenCV
        s8 = HyperSpyReader.to_uint8(signal)
        img = s8.data

        # CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip,
            tileGridSize=self.config.clahe_tile
        )
        enhanced = clahe.apply(img)

        # Black-hat background removal
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config.background_kernel, self.config.background_kernel)
        )
        bg_removed = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

        # Denoising
        if self.config.filter_type == 'bilateral':
            denoised = cv2.bilateralFilter(bg_removed, 9, 75, 75)
        else:
            denoised = cv2.GaussianBlur(bg_removed, (3, 3), 0)

        # Create new Signal2D preserving calibration
        result = signal.deepcopy()
        result.data = denoised
        return result

    def _report_results(self, base_name, aerosols, original_signal):
        """Generate reports for all detected aerosols."""
        # Labeled mask
        full_shape = original_signal.data.shape
        labeled_mask = np.zeros(full_shape, dtype=np.uint16)
        for i, obj in enumerate(aerosols):
            x, y, w, h = obj.bbox
            mask_global = np.zeros(full_shape, dtype=np.uint8)
            mask_global[y:y + h, x:x + w] = obj.mask_roi
            labeled_mask[mask_global > 0] = i + 1

        self.reporter.save_labeled_mask(base_name, labeled_mask, len(aerosols))

        # Individual results
        pix2nm = 1.0
        try:
            pix2nm = HyperSpyReader.get_scale_nm(original_signal)
        except ValueError:
            pass

        summary_data = []
        for obj in aerosols:
            self.reporter.save_aerosol_results(base_name, obj, pix2nm)
            summary_data.append({"id": obj.aerosol_id, "metrics": obj.metrics})

        if summary_data:
            self.reporter.export_summary(base_name, summary_data, pix2nm)
