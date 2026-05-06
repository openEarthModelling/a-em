"""Debug TraditionalCV failure and tune SAM parameters."""
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

# Debug TraditionalCV
print("=== TraditionalCV Debug ===")
trad_config = PipelineConfig(particle_type='soot')
trad = TraditionalCVSegmenter()

# Step through TraditionalCV manually
from a_em.io import HyperSpyReader
from scipy import ndimage

uint8 = HyperSpyReader.to_uint8(signal)
gray = uint8.data

print(f"uint8 min/max: {gray.min()} / {gray.max()}")

# Otsu threshold
from skimage.filters import threshold_otsu
if np.isnan(gray).any():
    gray = np.nan_to_num(gray)
try:
    thresh = threshold_otsu(gray)
except ValueError as e:
    print(f"Otsu failed: {e}")
    thresh = None

if thresh is not None:
    print(f"Otsu threshold: {thresh}")
    binary = gray < thresh  # particles are dark
    print(f"Binary foreground: {np.count_nonzero(binary)} pixels")
    
    # Morphological opening
    from skimage.morphology import disk, opening
    cleaned = opening(binary, disk(2))
    print(f"After opening: {np.count_nonzero(cleaned)} pixels")
    
    # Remove small objects
    from skimage.morphology import remove_small_objects
    labeled = ndimage.label(cleaned)[0]
    print(f"Number of labeled regions: {labeled.max()}")

# Try multiple SAM configs
print("\n=== SAM Parameter Sweep ===")
configs = [
    ('default', PipelineConfig(
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
    )),
    ('loose_iou', PipelineConfig(
        particle_type='soot',
        sam_checkpoint_path=CHECKPOINT,
        sam_device='auto',
        sam_points_per_side=32,
        sam_pred_iou_thresh=0.70,
        sam_stability_score_thresh=0.90,
        sam_min_area_ratio=0.0001,
        sam_max_area_ratio=0.70,
        sam_intensity_ratio=0.90,
        sam_edge_margin=3,
    )),
    ('dense_points', PipelineConfig(
        particle_type='soot',
        sam_checkpoint_path=CHECKPOINT,
        sam_device='auto',
        sam_points_per_side=64,
        sam_pred_iou_thresh=0.80,
        sam_stability_score_thresh=0.92,
        sam_min_area_ratio=0.0001,
        sam_max_area_ratio=0.70,
        sam_intensity_ratio=0.90,
        sam_edge_margin=3,
    )),
]

fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Original
axes[0, 0].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Traditional
config0 = PipelineConfig(particle_type='soot')
trad_mask = trad.segment(signal, config0)
axes[0, 1].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
axes[0, 1].imshow(trad_mask, cmap='Reds', alpha=0.4)
axes[0, 1].set_title(f'TraditionalCV ({np.count_nonzero(trad_mask)} px)')
axes[0, 1].axis('off')

for idx, (name, cfg) in enumerate(configs):
    sam = SAMAutoMaskSegmenter()
    mask = sam.segment(signal, cfg)
    print(f"\n{name}:")
    print(f"  Foreground pixels: {np.count_nonzero(mask)} ({np.count_nonzero(mask)/mask.size*100:.3f}%)")
    
    row = 0 if idx < 2 else 1
    col = 2 + (idx % 2)
    
    axes[row, col].imshow(img, cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    axes[row, col].imshow(mask, cmap='Greens', alpha=0.4)
    axes[row, col].set_title(f'SAM: {name}\n({np.count_nonzero(mask)} px)')
    axes[row, col].axis('off')

# Fill remaining
for r in range(2):
    for c in range(4):
        if not axes[r, c].has_data():
            axes[r, c].axis('off')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'sam_param_sweep_40026.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
