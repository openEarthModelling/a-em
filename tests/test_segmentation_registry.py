import pytest
from a_em.segmentation.base import SegmentationBackend
from a_em.segmentation.registry import SegmentationRegistry
from a_em.segmentation.sam_automask import SAMAutoMaskSegmenter


class DummyBackend(SegmentationBackend):
    name = 'dummy'
    def segment(self, signal, config):
        return signal.data


@pytest.fixture(autouse=True)
def clear_registry():
    SegmentationRegistry.clear()
    yield
    SegmentationRegistry.clear()


def test_register_and_get():
    SegmentationRegistry.register(DummyBackend)
    backend = SegmentationRegistry.get('dummy')
    assert isinstance(backend, DummyBackend)


def test_get_unknown_raises():
    with pytest.raises(ValueError, match="Unknown segmentation backend"):
        SegmentationRegistry.get('nonexistent')


def test_list_backends():
    assert SegmentationRegistry.list_backends() == []
    SegmentationRegistry.register(DummyBackend)
    assert SegmentationRegistry.list_backends() == ['dummy']


def test_find_compatible():
    class SootOnly(SegmentationBackend):
        name = 'soot_only'
        @classmethod
        def supports(cls, mt, pt):
            return pt == 'soot'
        def segment(self, signal, config):
            return signal.data
    SegmentationRegistry.register(SootOnly)
    assert SegmentationRegistry.find_compatible('TEM', 'soot') == 'soot_only'
    with pytest.raises(ValueError):
        SegmentationRegistry.find_compatible('TEM', 'spherical')


def test_register_duplicate_name_raises():
    SegmentationRegistry.register(DummyBackend)

    class AnotherDummy(SegmentationBackend):
        name = 'dummy'
        def segment(self, signal, config):
            return signal.data

    with pytest.raises(ValueError, match="already registered"):
        SegmentationRegistry.register(AnotherDummy)


def test_register_same_class_again_ok():
    SegmentationRegistry.register(DummyBackend)
    SegmentationRegistry.register(DummyBackend)  # Should not raise
    assert SegmentationRegistry.list_backends() == ['dummy']


def test_register_without_name_raises():
    class NoName(SegmentationBackend):
        def segment(self, signal, config):
            return signal.data

    with pytest.raises(ValueError, match="must define a 'name'"):
        SegmentationRegistry.register(NoName)


def test_register_non_backend_raises():
    with pytest.raises(TypeError):
        SegmentationRegistry.register(str)


def test_sam_automask_registered():
    SegmentationRegistry.register(SAMAutoMaskSegmenter)
    backend = SegmentationRegistry.get('sam_automask')
    assert isinstance(backend, SAMAutoMaskSegmenter)


def test_sam_automask_supports_all():
    assert SAMAutoMaskSegmenter.supports('TEM', 'soot') is True
    assert SAMAutoMaskSegmenter.supports('SEM', 'spherical') is True
    assert SAMAutoMaskSegmenter.supports('AFM', 'nanorod') is True
