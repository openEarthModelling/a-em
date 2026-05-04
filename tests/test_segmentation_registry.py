import pytest
from atem_analyzer.segmentation.base import SegmentationBackend
from atem_analyzer.segmentation.registry import SegmentationRegistry


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


def test_register_non_backend_raises():
    with pytest.raises(TypeError):
        SegmentationRegistry.register(str)
