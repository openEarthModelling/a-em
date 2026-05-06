from a_em.segmentation import (
    SegmentationRegistry,
    TraditionalCVSegmenter,
)


def test_traditional_cv_registered():
    # Ensure registration in case a previous test cleared the registry
    SegmentationRegistry.register(TraditionalCVSegmenter)
    backends = SegmentationRegistry.list_backends()
    assert 'traditional_cv' in backends
