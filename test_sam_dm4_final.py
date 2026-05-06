"""Final SAM test with best parameters on 40026.dm4."""
import sys
import time
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

# Best SAM config from parameter sweep
sam_config = PipelineConfig(
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

print("Running SAM with best parameters...")
sam = SAMAutoMaskSegmenter()
t0 = time.time()
sam_mask = sam.segment(signal, sam_config)
t1 = time.time()
print(f"SAM time: {t1-t0:.2f}s")
print(f"SAM foreground pixels: {np.count_nonzero(sam_mask)} ({np.count_nonzero(sam_mask)/sam_mask.size*100:.3f}%)")

print("\nRunning TraditionalCV...")
trad_config = PipelineConfig(particle_type='soot')
trad = TraditionalCVSegmenter()
t0 = time.time()
trad_mask = trad.segment(signal, trad_config)
t1 = time.time()
print(f"TraditionalCV time: {t1-t0:.2f}s")
print(f"TraditionalCV foreground pixels: {np.count_nonzero(trad_mask)} ({np.count_nonzero(trad_mask)/trad_mask.size*100:.3f}%)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Original
axes[0, 0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[0, 0].set_title('Original DM4')
axes[0, 0].axis('off')

# SAM mask only
axes[0, 1].imshow(sam_mask, cmap='gray')
axes[0, 1].set_title(f'SAM Mask\n({np.count_nonzero(sam_mask)} px)')
axes[0, 1].axis('off')

# Traditional mask only
axes[0, 2].imshow(trad_mask, cmap='gray')
axes[0, 2].set_title(f'Traditional Mask\n({np.count_nonzero(trad_mask)} px)')
axes[0, 2].axis('off')

# Original + SAM overlay
axes[1, 0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[1, 0].imshow(sam_mask, cmap='Greens', alpha=0.4)
axes[1, 0].set_title('SAM Overlay')
axes[1, 0].axis('off')

# Original + Traditional overlay
axes[1, 1].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[1, 1].imshow(trad_mask, cmap='Reds', alpha=0.4)
axes[1, 1].set_title('TraditionalCV Overlay')
axes[1, 1].axis('off')

# Difference: SAM only (green), Trad only (red), Both (yellow)
diff = np.zeros((*sam_mask.shape, 3))
sam_fg = sam_mask > 0
trad_fg = trad_mask > 0
both = np.logical_and(sam_fg, trad_fg)
sam_only = np.logical_and(sam_fg, ~trad_fg)
trad_only = np.logical_and(~sam_fg, trad_fg)

diff[sam_only] = [0, 1, 0]      # Green
diff[trad_only] = [1, 0, 0]     # Red
diff[both] = [1, 1, 0]          # Yellow
axes[1, 2].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[1, 2].imshow(diff, alpha=0.5)
axes[1, 2].set_title(f'Difference\nGreen=SAM only, Red=Trad only, Yellow=Both')
axes[1, 2].axis('off')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'final_comparison_40026.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")

# Save masks
np.save(os.path.join(OUTPUT_DIR, 'sam_best_mask.npy'), sam_mask)
print("Test completed successfully!")
