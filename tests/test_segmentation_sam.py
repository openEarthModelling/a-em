"""Unit tests for SAMAutoMaskSegmenter backend."""
import numpy as np
import pytest
import sys
from unittest.mock import MagicMock, patch

from atem_analyzer.config import PipelineConfig
from atem_analyzer.segmentation.sam_automask import SAMAutoMaskSegmenter


@pytest.fixture
def config():
    return PipelineConfig(
        sam_checkpoint_path='/fake/sam_vit_b.pth',
        sam_device='cpu',
        sam_points_per_side=32,
        sam_pred_iou_thresh=0.88,
        sam_stability_score_thresh=0.95,
        sam_min_area_ratio=0.0005,
        sam_max_area_ratio=0.60,
        sam_intensity_ratio=0.85,
        sam_edge_margin=5,
    )


@pytest.fixture
def fake_signal():
    """Return a mock signal with 100x100 uint8 data (bright background, one dark pixel)."""
    sig = MagicMock()
    sig.data = np.ones((100, 100), dtype=np.uint8) * 200
    sig.data[50, 50] = 50  # dark pixel for mask region
    return sig


class TestSAMAutoMaskSegmenter:

    def test_name(self):
        assert SAMAutoMaskSegmenter.name == 'sam_automask'

    def test_supports_soot(self):
        assert SAMAutoMaskSegmenter.supports('TEM', 'soot') is True
        assert SAMAutoMaskSegmenter.supports('SEM', 'soot') is True

    def test_supports_any_particle_type(self):
        assert SAMAutoMaskSegmenter.supports('TEM', 'spherical') is True
        assert SAMAutoMaskSegmenter.supports('SEM', 'nanorod') is True
        assert SAMAutoMaskSegmenter.supports('AFM', 'soot') is True

    def test_lazy_loading(self, config):
        segmenter = SAMAutoMaskSegmenter()
        assert segmenter._sam is None
        assert segmenter._mask_generator is None

    def test_segment_returns_binary_mask(self, config, fake_signal):
        segmenter = SAMAutoMaskSegmenter()

        mock_sam = MagicMock()
        mock_generator = MagicMock()
        mock_generator.generate.return_value = [
            {
                'segmentation': np.zeros((100, 100), dtype=bool),
                'area': 500,
                'bbox': [10, 10, 20, 20],
                'predicted_iou': 0.95,
                'stability_score': 0.97,
            }
        ]
        # Set one pixel so mask is non-zero after filtering
        mock_generator.generate.return_value[0]['segmentation'][50, 50] = True

        fake_module = MagicMock()
        fake_module.sam_model_registry = {'vit_b': MagicMock(return_value=mock_sam)}
        fake_module.SamAutomaticMaskGenerator = MagicMock(return_value=mock_generator)

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {'segment_anything': fake_module, 'torch': fake_torch}):
            with patch('atem_analyzer.io.HyperSpyReader.to_uint8', return_value=fake_signal):
                with patch('os.path.exists', return_value=True):
                    mask = segmenter.segment(fake_signal, config)

        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})
        assert mask.shape == (100, 100)
        # The one True pixel should be 255
        assert mask[50, 50] == 255

    def test_segment_zero_mask_when_no_masks_pass_filter(self, config, fake_signal):
        segmenter = SAMAutoMaskSegmenter()

        mock_sam = MagicMock()
        mock_generator = MagicMock()
        # Mask that fails area filter (too small)
        mock_generator.generate.return_value = [
            {
                'segmentation': np.zeros((100, 100), dtype=bool),
                'area': 1,
                'bbox': [10, 10, 2, 2],
                'predicted_iou': 0.95,
                'stability_score': 0.97,
            }
        ]

        fake_module = MagicMock()
        fake_module.sam_model_registry = {'vit_b': MagicMock(return_value=mock_sam)}
        fake_module.SamAutomaticMaskGenerator = MagicMock(return_value=mock_generator)

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {'segment_anything': fake_module, 'torch': fake_torch}):
            with patch('atem_analyzer.io.HyperSpyReader.to_uint8', return_value=fake_signal):
                with patch('os.path.exists', return_value=True):
                    mask = segmenter.segment(fake_signal, config)

        assert mask.dtype == np.uint8
        assert np.all(mask == 0)

    def test_missing_checkpoint_raises_filenotfound(self, config, fake_signal):
        segmenter = SAMAutoMaskSegmenter()
        config.sam_checkpoint_path = ''

        fake_module = MagicMock()
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {'segment_anything': fake_module, 'torch': fake_torch}):
            with pytest.raises(FileNotFoundError, match='SAM checkpoint not found'):
                segmenter.segment(fake_signal, config)

    def test_missing_segment_anything_raises_importerror(self, config, fake_signal):
        segmenter = SAMAutoMaskSegmenter()

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False

        # Ensure segment_anything is not in sys.modules
        with patch.dict(sys.modules, {'segment_anything': None, 'torch': fake_torch}, clear=False):
            # Remove segment_anything if it exists
            if 'segment_anything' in sys.modules:
                del sys.modules['segment_anything']
            with pytest.raises(ImportError):
                segmenter.segment(fake_signal, config)
