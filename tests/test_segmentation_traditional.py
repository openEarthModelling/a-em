import numpy as np
import pytest
import hyperspy.api as hs

from a_em.segmentation.traditional import TraditionalCVSegmenter
from a_em.config import PipelineConfig


@pytest.fixture
def segmenter():
    return TraditionalCVSegmenter()


@pytest.fixture
def fake_signal():
    data = np.ones((200, 200), dtype=np.float32) * 200
    data[50:150, 50:150] = 50.0
    s = hs.signals.Signal2D(data)
    s.axes_manager[0].scale = 1.0
    s.axes_manager[0].units = 'nm'
    s.axes_manager[1].scale = 1.0
    s.axes_manager[1].units = 'nm'
    return s


def test_supports_soot(segmenter):
    assert TraditionalCVSegmenter.supports('TEM', 'soot') is True
    assert TraditionalCVSegmenter.supports('SEM', 'soot') is True


def test_does_not_support_spherical(segmenter):
    assert TraditionalCVSegmenter.supports('TEM', 'spherical') is False


def test_segment_produces_mask(segmenter, fake_signal):
    config = PipelineConfig(min_area=50, use_grabcut_refinement=False)
    mask = segmenter.segment(fake_signal, config)
    assert mask.dtype == np.uint8
    assert mask.shape == fake_signal.data.shape
    assert set(np.unique(mask)).issubset({0, 255})


def test_extract_objects(segmenter, fake_signal):
    config = PipelineConfig(min_area=50, use_grabcut_refinement=False)
    mask = segmenter.segment(fake_signal, config)
    objects = segmenter.extract_objects(fake_signal, mask, config)
    assert len(objects) >= 1
    for obj in objects:
        assert obj.area >= config.min_area
