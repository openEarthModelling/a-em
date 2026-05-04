from atem_analyzer.segmentation import SegmentationRegistry


def test_traditional_cv_registered():
    backends = SegmentationRegistry.list_backends()
    assert 'traditional_cv' in backends
