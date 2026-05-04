import pytest
from atem_analyzer.segmentation.base import SegmentationBackend


def test_segmentation_backend_is_abstract():
    with pytest.raises(TypeError):
        SegmentationBackend()
