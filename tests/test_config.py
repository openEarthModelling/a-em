from atem_analyzer.config import PipelineConfig


def test_default_config():
    cfg = PipelineConfig()
    assert cfg.microscope_type == 'TEM'
    assert cfg.particle_type == 'soot'
    assert cfg.segmentation_backend == 'traditional_cv'
    assert cfg.analysis_engine == 'soot'
    assert cfg.clahe_clip == 3.0


def test_config_override():
    cfg = PipelineConfig(microscope_type='SEM', particle_type='spherical')
    assert cfg.microscope_type == 'SEM'
    assert cfg.particle_type == 'spherical'
