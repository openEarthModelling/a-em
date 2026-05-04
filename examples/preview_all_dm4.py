"""Generate preview montages for all DM4 files under a directory.

Usage:
    python preview_all_dm4.py
"""
from pathlib import Path

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = '/data/zhangfan_data/pangyuner'
OUTPUT_DIR = Path(ROOT_DIR) / 'preview'
OUTPUT_DIR.mkdir(exist_ok=True)

THUMB_WIDTH = 400
NCOLS = 4


def make_thumbnail(signal, width=THUMB_WIDTH):
    """Resize signal data to target width while keeping aspect ratio."""
    data = signal.data.astype(float)
    data = (data - data.min()) / (data.max() - data.min() + 1e-9)

    h, w = data.shape
    scale = width / w
    new_h, new_w = int(h * scale), width

    # Use cv2 if available, otherwise simple numpy resize
    try:
        import cv2
        thumb = cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        thumb = np.array([
            [data[int(y / scale), int(x / scale)] for x in range(new_w)]
            for y in range(new_h)
        ])
    return thumb


def build_montage(image_list, ncols=NCOLS):
    """Build a grid montage from a list of (thumb, title) tuples."""
    n = len(image_list)
    if n == 0:
        return None

    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (thumb, title) in enumerate(image_list):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.imshow(thumb, cmap='gray')
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig


def process_directory(subdir: Path):
    """Process all DM4 files in a subdirectory."""
    dm4_files = sorted(subdir.glob('*.dm4'))
    if not dm4_files:
        return

    print(f"[*] {subdir.name}: {len(dm4_files)} files")
    images = []

    for f in dm4_files:
        try:
            signal = hs.load(str(f))
            if isinstance(signal, list):
                signal = signal[0]
            thumb = make_thumbnail(signal)
            images.append((thumb, f.name))
        except Exception as e:
            print(f"  [!] Skip {f.name}: {e}")

    if not images:
        return

    fig = build_montage(images, ncols=NCOLS)
    if fig is None:
        return

    out_path = OUTPUT_DIR / f"{subdir.name}_preview.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved: {out_path}")


if __name__ == '__main__':
    for subdir in sorted(Path(ROOT_DIR).iterdir()):
        if subdir.is_dir() and subdir.name != 'preview':
            process_directory(subdir)

    print(f"\nDone. Previews saved to: {OUTPUT_DIR}")
