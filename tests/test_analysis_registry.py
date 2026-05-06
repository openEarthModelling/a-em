import pytest
from a_em.analysis.base import AnalysisEngine
from a_em.analysis.registry import AnalysisEngineRegistry


class DummyEngine(AnalysisEngine):
    name = 'dummy'

    def analyze(self, signal_roi, mask_roi):
        return {'value': 42}


@pytest.fixture(autouse=True)
def clear_registry():
    AnalysisEngineRegistry.clear()
    yield
    AnalysisEngineRegistry.clear()


def test_register_and_get():
    AnalysisEngineRegistry.register(DummyEngine)
    engine = AnalysisEngineRegistry.get('dummy')
    assert isinstance(engine, DummyEngine)


def test_get_unknown_raises():
    with pytest.raises(ValueError, match="Unknown analysis engine"):
        AnalysisEngineRegistry.get('nonexistent')


def test_list_engines():
    assert AnalysisEngineRegistry.list_engines() == []
    AnalysisEngineRegistry.register(DummyEngine)
    assert AnalysisEngineRegistry.list_engines() == ['dummy']


def test_find_compatible():
    class SootOnly(AnalysisEngine):
        name = 'soot_only'
        @classmethod
        def supports(cls, pt):
            return pt == 'soot'
        def analyze(self, signal_roi, mask_roi):
            return {}
    AnalysisEngineRegistry.register(SootOnly)
    assert AnalysisEngineRegistry.find_compatible('soot') == 'soot_only'
    with pytest.raises(ValueError):
        AnalysisEngineRegistry.find_compatible('spherical')


def test_register_duplicate_name_raises():
    AnalysisEngineRegistry.register(DummyEngine)

    class AnotherDummy(AnalysisEngine):
        name = 'dummy'
        def analyze(self, signal_roi, mask_roi):
            return {}

    with pytest.raises(ValueError, match="already registered"):
        AnalysisEngineRegistry.register(AnotherDummy)


def test_register_same_class_again_ok():
    AnalysisEngineRegistry.register(DummyEngine)
    AnalysisEngineRegistry.register(DummyEngine)  # Should not raise
    assert AnalysisEngineRegistry.list_engines() == ['dummy']


def test_register_without_name_raises():
    class NoName(AnalysisEngine):
        def analyze(self, signal_roi, mask_roi):
            return {}

    with pytest.raises(ValueError, match="must define a 'name'"):
        AnalysisEngineRegistry.register(NoName)


def test_register_non_engine_raises():
    with pytest.raises(TypeError):
        AnalysisEngineRegistry.register(str)
