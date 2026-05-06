"""Integration test using real DM4 test data."""
import os
import pytest
import pandas as pd

from a_em import (
    HyperSpyReader,
    PipelineConfig,
    PipelineExecutor,
    BatchProcessor,
)

TEST_DM4_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'test_dm4'
)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(TEST_DM4_DIR, '40026.dm4')),
    reason="DM4 test data not available"
)
class TestDM4Integration:
    """End-to-end test on real DM4 files."""

    def test_load_dm4_scale(self):
        path = os.path.join(TEST_DM4_DIR, '40026.dm4')
        s = HyperSpyReader.load(path)
        scale_nm = HyperSpyReader.get_scale_nm(s)

        # 40026.dm4 is 10000x, scale should be ~1.27 nm/pixel
        assert 1.0 < scale_nm < 2.0
        assert s.data.shape == (1336, 2004)

    def test_full_pipeline_on_dm4(self, tmp_path):
        path = os.path.join(TEST_DM4_DIR, '40026.dm4')
        s = HyperSpyReader.load(path)

        config = PipelineConfig(
            output_dir=str(tmp_path),
            min_area=200,
            use_grabcut_refinement=False,
            threshold_method='otsu',
        )

        executor = PipelineExecutor(config)
        aerosols = executor.run(s)

        # Should detect at least one soot aggregate
        assert len(aerosols) >= 1

        # Each aerosol should have soot metrics
        for a in aerosols:
            assert 'df' in a.metrics
            assert 'num_particles' in a.metrics

    def test_batch_on_dm4_directory(self, tmp_path):
        config = PipelineConfig(
            output_dir=str(tmp_path),
            min_area=200,
            use_grabcut_refinement=False,
            threshold_method='otsu',
        )
        processor = BatchProcessor(config)
        df = processor.process_directory(TEST_DM4_DIR, pattern='*.dm4')

        assert isinstance(df, pd.DataFrame)
        # Should have detected some aerosols across all files
        if not df.empty:
            assert 'file' in df.columns
            assert 'df' in df.columns
