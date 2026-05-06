"""Single-image soot aerosol analysis example."""
import matplotlib.pyplot as plt
import numpy as np

from a_em import (
    HyperSpyReader,
    PipelineConfig,
    AerosolReporter,
)
from a_em.pipeline import PipelineExecutor
from a_em.segmentation.registry import SegmentationRegistry
from a_em.analysis.registry import AnalysisEngineRegistry
import cv2

# ---------------------------------------------------------------------------
# 1. Load a DM4 TEM image
# ---------------------------------------------------------------------------
signal = HyperSpyReader.load('data/test_dm4/40026.dm4')
print(f"Loaded: {signal.metadata.General.title}")
print(f"Shape : {signal.data.shape}")

# ---------------------------------------------------------------------------
# 2. Configure the pipeline for soot particle analysis
# ---------------------------------------------------------------------------
config = PipelineConfig(
    microscope_type='TEM',
    particle_type='soot',
    min_area=100,
    output_dir='data/processed',
)

# ---------------------------------------------------------------------------
# 3. Run pipeline stages manually (to control what gets saved)
# ---------------------------------------------------------------------------
segmenter = SegmentationRegistry.get(config.segmentation_backend)
engine = AnalysisEngineRegistry.get(config.analysis_engine)
reporter = AerosolReporter(output_dir=config.output_dir)

# 3.1 Preprocess
executor = PipelineExecutor(config)
preprocessed = executor._preprocess(signal)

# 3.2 Segment
mask = segmenter.segment(preprocessed, config)

# 3.3 Extract objects
aerosols = segmenter.extract_objects(signal, mask, config)

# 3.4 Analyze
for obj in aerosols:
    metrics = engine.analyze(obj.image_roi, obj.mask_roi)
    obj.metrics = metrics

print(f"\nDetected {len(aerosols)} aerosol object(s)")
for obj in aerosols:
    print(f"  ID={obj.aerosol_id:3d}  area={obj.area:8.1f}px  metrics={list(obj.metrics.keys())}")

# 3.5 Save only the labeled mask (skip individual per-aerosol files)
base_name = signal.metadata.General.title
full_shape = signal.data.shape
labeled_mask = np.zeros(full_shape, dtype=np.uint16)
for i, obj in enumerate(aerosols):
    if obj.bbox is None:
        continue
    x, y, w, h = obj.bbox
    mask_global = np.zeros(full_shape, dtype=np.uint8)
    mask_global[y:y + h, x:x + w] = obj.mask_roi
    labeled_mask[mask_global > 0] = i + 1

mask_path = reporter.save_labeled_mask(base_name, labeled_mask, len(aerosols))
print(f"\nLabeled mask saved to: {mask_path}")

# ---------------------------------------------------------------------------
# 4. Plot results (original + detections)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original image
img = signal.data.astype(float)
img = (img - img.min()) / (img.max() - img.min())
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Detection overlay
overlay = np.stack([img] * 3, axis=-1)
overlay = (overlay * 255).astype(np.uint8)

for obj in aerosols:
    if obj.bbox is None:
        continue
    x, y, w, h = obj.bbox
    # Draw bounding box (red)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw primary particle circles if available
    particles = obj.metrics.get('primary_particles') or []
    for p in particles:
        cy, cx = p['center']
        r = int(p['radius'])
        cv2.circle(overlay, (cx, cy), r, (0, 255, 0), 1)
        cv2.circle(overlay, (cx, cy), 2, (0, 0, 255), -1)

axes[1].imshow(overlay)
axes[1].set_title(f'Detections: {len(aerosols)} aerosols')
axes[1].axis('off')

plt.tight_layout()
plot_path = 'data/processed/detection_result.png'
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to: {plot_path}")
plt.show()
