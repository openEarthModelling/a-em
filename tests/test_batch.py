import pandas as pd
import pytest
import hyperspy.api as hs
import numpy as np

from atem_analyzer.batch import BatchProcessor
from atem_analyzer.config import PipelineConfig
from atem_analyzer.segmentation.registry import SegmentationRegistry
from atem_analyzer.segmentation.traditional import TraditionalCVSegmenter
from atem_analyzer.analysis.registry import AnalysisEngineRegistry
from atem_analyzer.analysis.soot import SootAnalysisEngine


@pytest.fixture(autouse=True)
def ensure_backends_registered():
    """Re-register backends before each test (other tests may clear registries)."""
    SegmentationRegistry.register(TraditionalCVSegmenter)
    AnalysisEngineRegistry.register(SootAnalysisEngine)


@pytest.fixture
def temp_input_dir(tmp_path):
    """Create a temp directory with fake TIF files."""
    d = tmp_path / "input"
    d.mkdir()
    cv2 = pytest.importorskip('cv2')

    for i in range(3):
        data = np.ones((200, 200), dtype=np.float32) * 180
        cv2.circle(data, (100, 100), 30, 50.0, -1)
        cv2.circle(data, (115, 95), 20, 50.0, -1)
        cv2.circle(data, (85, 110), 18, 50.0, -1)
        s = hs.signals.Signal2D(data)
        s.axes_manager[0].scale = 1.0
        s.axes_manager[0].units = 'nm'
        s.axes_manager[1].scale = 1.0
        s.axes_manager[1].units = 'nm'
        s.save(str(d / f"test_{i:02d}.tif"))

    return d


class TestBatchProcessor:
    def test_process_directory(self, temp_input_dir, tmp_path):
        config = PipelineConfig(
            output_dir=str(tmp_path / "output"),
            min_area=50,
            use_grabcut_refinement=False
        )
        processor = BatchProcessor(config)
        df = processor.process_directory(str(temp_input_dir), pattern='*.tif')

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        assert 'file' in df.columns

    def test_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        processor = BatchProcessor()
        df = processor.process_directory(str(empty_dir), pattern='*.dm4')
        assert df.empty
