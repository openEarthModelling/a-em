from a_em.config import PipelineConfig


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


def test_sam_config_defaults():
    cfg = PipelineConfig()
    assert cfg.sam_model_type == 'vit_b'
    assert cfg.sam_checkpoint_path == ''
    assert cfg.sam_device == 'auto'
    assert cfg.sam_points_per_side == 32
    assert cfg.sam_pred_iou_thresh == 0.88
    assert cfg.sam_stability_score_thresh == 0.95
    assert cfg.sam_min_area_ratio == 0.0005
    assert cfg.sam_max_area_ratio == 0.60
    assert cfg.sam_intensity_ratio == 0.85
    assert cfg.sam_edge_margin == 5


def test_sam_config_override():
    cfg = PipelineConfig(sam_model_type='vit_h', sam_device='cuda')
    assert cfg.sam_model_type == 'vit_h'
    assert cfg.sam_device == 'cuda'
