"""Detailed analysis of SAM vs TraditionalCV on 40026.dm4."""
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

signal = HyperSpyReader.load(DM4_PATH)
img = signal.data

print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")
print(f"Min/Max: {img.min():.2f} / {img.max():.2f}")
print(f"Mean/Std: {img.mean():.2f} / {img.std():.2f}")
print(f"1st percentile: {np.percentile(img, 1):.2f}")
print(f"5th percentile: {np.percentile(img, 5):.2f}")
print(f"10th percentile: {np.percentile(img, 10):.2f}")
print(f"50th percentile: {np.percentile(img, 50):.2f}")
print(f"90th percentile: {np.percentile(img, 90):.2f}")
print(f"99th percentile: {np.percentile(img, 99):.2f}")

# Run both backends
config = PipelineConfig(
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

sam = SAMAutoMaskSegmenter()
sam_mask = sam.segment(signal, config)

trad = TraditionalCVSegmenter()
trad_mask = trad.segment(signal, config)

print(f"\nSAM mask pixels: {np.count_nonzero(sam_mask)} ({np.count_nonzero(sam_mask)/sam_mask.size*100:.3f}%)")
print(f"Trad mask pixels: {np.count_nonzero(trad_mask)} ({np.count_nonzero(trad_mask)/trad_mask.size*100:.3f}%)")

# Detailed figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Original + masks
axes[0, 0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[0, 0].set_title('Original (percentile scaled)')
axes[0, 0].axis('off')

axes[0, 1].imshow(trad_mask, cmap='gray')
axes[0, 1].set_title(f'TraditionalCV ({np.count_nonzero(trad_mask)} px)')
axes[0, 1].axis('off')

axes[0, 2].imshow(sam_mask, cmap='gray')
axes[0, 2].set_title(f'SAM ({np.count_nonzero(sam_mask)} px)')
axes[0, 2].axis('off')

# Row 2: Overlays
axes[1, 0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[1, 0].set_title('Original')
axes[1, 0].axis('off')

axes[1, 1].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[1, 1].imshow(trad_mask, cmap='Reds', alpha=0.4)
axes[1, 1].set_title('TraditionalCV overlay')
axes[1, 1].axis('off')

axes[1, 2].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[1, 2].imshow(sam_mask, cmap='Greens', alpha=0.4)
axes[1, 2].set_title('SAM overlay')
axes[1, 2].axis('off')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'comparison_detailed_40026.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
