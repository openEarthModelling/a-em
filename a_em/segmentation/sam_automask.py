"""SAM Auto-Mask segmentation backend with post-filtering."""
import logging
import os

import numpy as np

from a_em.segmentation.base import SegmentationBackend
from a_em.config import PipelineConfig

logger = logging.getLogger(__name__)

_CHECKPOINT_URLS = {
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
}

_DEFAULT_CHECKPOINT_NAMES = {
    'vit_h': 'sam_vit_h_4b8939.pth',
    'vit_l': 'sam_vit_l_0b3195.pth',
    'vit_b': 'sam_vit_b_01ec64.pth',
}


def _is_nested(small_mask, large_mask, threshold=0.8):
    """Check if small_mask is mostly contained within large_mask."""
    intersection = np.logical_and(small_mask, large_mask).sum()
    containment = intersection / (small_mask.sum() + 1e-8)
    return bool(containment > threshold)


def post_filter_masks(masks, original_gray, image_shape, config):
    """Apply 5-stage post-filtering to SAM AMG output.

    Args:
        masks: List of dicts from AMG, each with 'segmentation' (bool HxW),
               'area', 'bbox' [x, y, w, h], 'predicted_iou', 'stability_score'.
        original_gray: Original grayscale image (H, W) uint8, before enhancement.
        image_shape: (H, W) of the image.
        config: PipelineConfig with SAM filter parameters.

    Returns:
        List of filtered mask dicts.
    """
    h, w = image_shape
    image_area = h * w
    filtered = []

    # Stage 1: Confidence filtering
    for mask in masks:
        if mask.get('predicted_iou', 0) < config.sam_pred_iou_thresh:
            continue
        if mask.get('stability_score', 0) < config.sam_stability_score_thresh:
            continue
        filtered.append(mask)

    # Stage 2: Area filtering
    min_area = image_area * config.sam_min_area_ratio
    max_area = image_area * config.sam_max_area_ratio
    filtered = [m for m in filtered if min_area <= m['area'] <= max_area]

    # Stage 3: Intensity filtering
    global_mean = original_gray.mean()
    intensity_passed = []
    for mask in filtered:
        seg = mask['segmentation']
        mean_masked = original_gray[seg].mean()
        if mean_masked < global_mean * config.sam_intensity_ratio:
            intensity_passed.append(mask)
    filtered = intensity_passed

    # Stage 4: Nested mask resolution
    sorted_by_area = sorted(filtered, key=lambda m: m['area'], reverse=True)
    selected = []
    for mask in sorted_by_area:
        seg = mask['segmentation']
        if any(_is_nested(seg, sel['segmentation']) for sel in selected):
            continue
        selected.append(mask)
    filtered = selected

    # Stage 5: Edge exclusion
    edge_margin = config.sam_edge_margin
    edge_passed = []
    for mask in filtered:
        x, y, bw, bh = mask['bbox']
        if (x < edge_margin or y < edge_margin or
                (x + bw) > (w - edge_margin) or (y + bh) > (h - edge_margin)):
            continue
        edge_passed.append(mask)

    return edge_passed


class SAMAutoMaskSegmenter(SegmentationBackend):
    """SAM Auto-Mask Generator segmentation backend with post-filtering."""

    name: str = 'sam_automask'

    def __init__(self):
        self._sam = None
        self._mask_generator = None

    @classmethod
    def supports(cls, microscope_type: str, particle_type: str) -> bool:
        # Decoupled from particle type
        return True

    def _load_model(self, config: PipelineConfig):
        if self._sam is not None:
            return

        try:
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError as e:
            raise ImportError(
                "segment-anything is required. "
                "Install with: pip install a_em[sam]"
            ) from e

        device = config.sam_device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint_path = config.sam_checkpoint_path
        if not checkpoint_path:
            checkpoint_path = _DEFAULT_CHECKPOINT_NAMES.get(
                config.sam_model_type, 'sam_vit_b_01ec64.pth'
            )

        if not os.path.exists(checkpoint_path):
            url = _CHECKPOINT_URLS.get(config.sam_model_type, _CHECKPOINT_URLS['vit_b'])
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}. "
                f"Download from: {url}"
            )

        logger.info("Loading SAM model %s on %s", config.sam_model_type, device)

        sam = sam_model_registry[config.sam_model_type](checkpoint=checkpoint_path)
        sam.to(device=device)

        self._sam = sam
        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=config.sam_points_per_side,
            pred_iou_thresh=config.sam_pred_iou_thresh,
            stability_score_thresh=config.sam_stability_score_thresh,
            min_mask_region_area=100,
        )

    def segment(self, signal, config: PipelineConfig) -> np.ndarray:
        from a_em.io import HyperSpyReader

        self._load_model(config)

        uint8_signal = HyperSpyReader.to_uint8(signal)
        gray = uint8_signal.data
        h, w = gray.shape[:2]

        if gray.ndim == 2:
            rgb = np.stack([gray, gray, gray], axis=-1)
        elif gray.ndim == 3 and gray.shape[2] == 1:
            rgb = np.repeat(gray, 3, axis=-1)
        else:
            rgb = gray

        logger.info("Running SAM AMG on image %s", rgb.shape[:2])
        masks = self._mask_generator.generate(rgb)
        logger.info("SAM generated %d candidate masks", len(masks))

        filtered = post_filter_masks(
            masks,
            original_gray=gray,
            image_shape=(h, w),
            config=config,
        )

        logger.info("Post-filtering retained %d masks", len(filtered))

        if not filtered:
            logger.warning("No masks passed post-filtering; returning empty mask")
            return np.zeros((h, w), dtype=np.uint8)

        binary_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in filtered:
            binary_mask[mask['segmentation']] = 255

        return binary_mask
