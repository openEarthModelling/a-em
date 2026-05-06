"""Example: Segment 40026.dm4 using SAM AutoMask.

Outputs to: data/processed/
"""
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/zhangfan/Project/20260319_SPEMBSSBDART/atem_analyzer')

from atem_analyzer.io import HyperSpyReader
from atem_analyzer.config import PipelineConfig
from atem_analyzer.segmentation.sam_automask import SAMAutoMaskSegmenter

# ── Paths ───────────────────────────────────────────────────────
DM4_PATH = '/home/zhangfan/Project/20260319_SPEMBSSBDART/atem_analyzer/data/test_dm4/40026.dm4'
CHECKPOINT = '/data/zhangfan_data/sam_vit_b_01ec64.pth'
OUTPUT_DIR = '/home/zhangfan/Project/20260319_SPEMBSSBDART/atem_analyzer/data/processed'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load image ──────────────────────────────────────────────────
print("Loading DM4 file...")
signal = HyperSpyReader.load(DM4_PATH)
img = signal.data
print(f"  Shape: {img.shape}, dtype: {img.dtype}")
print(f"  Range: {img.min():.2f} ~ {img.max():.2f}")

# ── Configure SAM ───────────────────────────────────────────────
# Best parameters from parameter sweep on 40026.dm4
config = PipelineConfig(
    particle_type='soot',
    sam_checkpoint_path=CHECKPOINT,
    sam_device='auto',
    sam_model_type='vit_b',
    sam_points_per_side=64,
    sam_pred_iou_thresh=0.80,
    sam_stability_score_thresh=0.92,
    sam_min_area_ratio=0.0001,
    sam_max_area_ratio=0.70,
    sam_intensity_ratio=0.90,
    sam_edge_margin=3,
)

# ── Run segmentation ────────────────────────────────────────────
print("\nRunning SAM AutoMask segmentation...")
segmenter = SAMAutoMaskSegmenter()
t0 = time.time()
mask = segmenter.segment(signal, config)
t1 = time.time()

print(f"  Done in {t1 - t0:.2f}s")
print(f"  Foreground pixels: {np.count_nonzero(mask)} ({np.count_nonzero(mask)/mask.size*100:.3f}%)")

# ── Save results ────────────────────────────────────────────────
# 1. Binary mask as .npy
mask_path = os.path.join(OUTPUT_DIR, '40026_sam_mask.npy')
np.save(mask_path, mask)
print(f"  Mask saved: {mask_path}")

# 2. Binary mask as .png
mask_png_path = os.path.join(OUTPUT_DIR, '40026_sam_mask.png')
plt.imsave(mask_png_path, mask, cmap='gray')
print(f"  Mask PNG saved: {mask_png_path}")

# 3. Overlay visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title(f'SAM Mask\n({np.count_nonzero(mask)} px)')
axes[1].axis('off')

axes[2].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[2].imshow(mask, cmap='Greens', alpha=0.4)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
overlay_path = os.path.join(OUTPUT_DIR, '40026_sam_overlay.png')
fig.savefig(overlay_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Overlay saved: {overlay_path}")

print("\nAll outputs written to:", OUTPUT_DIR)
