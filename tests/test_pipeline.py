import numpy as np
import pytest
import hyperspy.api as hs

from a_em.pipeline import PipelineExecutor
from a_em.config import PipelineConfig
from a_em.segmentation.registry import SegmentationRegistry
from a_em.segmentation.traditional import TraditionalCVSegmenter
from a_em.analysis.registry import AnalysisEngineRegistry
from a_em.analysis.soot import SootAnalysisEngine


@pytest.fixture(autouse=True)
def ensure_backends_registered():
    """Re-register backends before each test (other tests may clear registries)."""
    SegmentationRegistry.register(TraditionalCVSegmenter)
    AnalysisEngineRegistry.register(SootAnalysisEngine)


@pytest.fixture
def fake_soot_signal():
    """Create a fake signal with a dark aggregate-like region."""
    data = np.ones((200, 200), dtype=np.float32) * 180
    cv2 = pytest.importorskip('cv2')
    cv2.circle(data, (100, 100), 30, 50.0, -1)
    cv2.circle(data, (115, 95), 20, 50.0, -1)
    cv2.circle(data, (85, 110), 18, 50.0, -1)

    s = hs.signals.Signal2D(data)
    s.axes_manager[0].scale = 1.0
    s.axes_manager[0].units = 'nm'
    s.axes_manager[1].scale = 1.0
    s.axes_manager[1].units = 'nm'
    s.metadata.General.title = 'test_aggregate'
    return s


class TestPipelineExecutor:
    def test_pipeline_runs(self, fake_soot_signal, tmp_path):
        config = PipelineConfig(
            output_dir=str(tmp_path),
            min_area=50,
            use_grabcut_refinement=False
        )
        executor = PipelineExecutor(config)
        aerosols = executor.run(fake_soot_signal)

        assert isinstance(aerosols, list)
        assert len(aerosols) >= 1

    def test_pipeline_creates_outputs(self, fake_soot_signal, tmp_path):
        config = PipelineConfig(
            output_dir=str(tmp_path),
            min_area=50,
            use_grabcut_refinement=False
        )
        executor = PipelineExecutor(config)
        executor.run(fake_soot_signal)

        files = list(tmp_path.glob('*'))
        assert len(files) > 0
