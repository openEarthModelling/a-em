"""Test SAM backend on 40026.dm4 DM4 file."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/zhangfan/Project/20260319_SPEMBSSBDART/a_em')

from a_em.io import HyperSpyReader
from a_em.config import PipelineConfig
from a_em.segmentation.sam_automask import SAMAutoMaskSegmenter
from a_em.segmentation.traditional import TraditionalCVSegmenter

DM4_PATH = '/home/zhangfan/Project/20260319_SPEMBSSBDART/a_em/data/test_dm4/40026.dm4'
CHECKPOINT = '/data/zhangfan_data/sam_vit_b_01ec64.pth'
OUTPUT_DIR = '/home/zhangfan/Project/20260319_SPEMBSSBDART/a_em/test_output'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Loading DM4 file...")
signal = HyperSpyReader.load(DM4_PATH)
print(f"Signal shape: {signal.data.shape}")
print(f"Signal dtype: {signal.data.dtype}")
print(f"Signal min/max: {signal.data.min():.2f} / {signal.data.max():.2f}")

# --- Traditional CV ---
print("\n" + "=" * 60)
print("Running TraditionalCVSegmenter...")
trad_config = PipelineConfig(
    segmentation_backend='traditional_cv',
    particle_type='soot',
)
trad_segmenter = TraditionalCVSegmenter()
trad_mask = trad_segmenter.segment(signal, trad_config)
print(f"Traditional mask: shape={trad_mask.shape}, unique={np.unique(trad_mask)}")

# --- SAM Auto-Mask ---
print("\n" + "=" * 60)
print("Running SAMAutoMaskSegmenter...")
sam_config = PipelineConfig(
    segmentation_backend='sam_automask',
    particle_type='soot',
    sam_checkpoint_path=CHECKPOINT,
    sam_device='auto',
    sam_points_per_side=32,
    sam_pred_iou_thresh=0.88,
    sam_stability_score_thresh=0.95,
    sam_min_area_ratio=0.0005,
    sam_max_area_ratio=0.60,
    sam_intensity_ratio=0.85,
    sam_edge_margin=5,
)
sam_segmenter = SAMAutoMaskSegmenter()
sam_mask = sam_segmenter.segment(signal, sam_config)
print(f"SAM mask: shape={sam_mask.shape}, unique={np.unique(sam_mask)}")

# --- Visualization ---
print("\n" + "=" * 60)
print("Saving comparison figure...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original image
img = signal.data
if img.ndim == 3 and img.shape[2] == 1:
    img = img[:, :, 0]
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original DM4')
axes[0].axis('off')

# Traditional mask overlay
axes[1].imshow(img, cmap='gray')
axes[1].imshow(trad_mask, cmap='Reds', alpha=0.4)
axes[1].set_title(f'TraditionalCV (pixels={np.count_nonzero(trad_mask)})')
axes[1].axis('off')

# SAM mask overlay
axes[2].imshow(img, cmap='gray')
axes[2].imshow(sam_mask, cmap='Greens', alpha=0.4)
axes[2].set_title(f'SAM AutoMask (pixels={np.count_nonzero(sam_mask)})')
axes[2].axis('off')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'comparison_40026.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")

# Save individual masks
np.save(os.path.join(OUTPUT_DIR, 'traditional_mask.npy'), trad_mask)
np.save(os.path.join(OUTPUT_DIR, 'sam_mask.npy'), sam_mask)
print(f"Saved masks to {OUTPUT_DIR}")

print("\n" + "=" * 60)
print("Done!")
