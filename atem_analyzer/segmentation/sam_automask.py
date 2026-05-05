"""SAM Auto-Mask segmentation backend with post-filtering."""
import numpy as np


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
