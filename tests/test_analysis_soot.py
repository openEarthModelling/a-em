import numpy as np
import pytest

from atem_analyzer.analysis.soot import SootAnalysisEngine


@pytest.fixture
def engine():
    return SootAnalysisEngine()


def test_supports_soot(engine):
    assert SootAnalysisEngine.supports('soot') is True
    assert SootAnalysisEngine.supports('spherical') is False


def test_analyze_aggregate(engine):
    """Test on a synthetic aggregate-like shape."""
    cv2 = pytest.importorskip('cv2')

    # Create a mask with a few connected blobs
    mask = np.zeros((100, 100), dtype=np.uint8)

    # Draw overlapping circles to simulate aggregate
    for cx, cy, r in [(30, 30, 10), (45, 35, 8), (35, 50, 9), (55, 45, 7)]:
        cv2.circle(mask, (cx, cy), r, 255, -1)

    img = np.ones((100, 100), dtype=np.float32) * 200

    metrics = engine.analyze(img, mask)

    assert 'df' in metrics
    assert 'rg' in metrics
    assert 'r0' in metrics
    assert 'num_particles' in metrics
    assert metrics['num_particles'] >= 2
    assert 'primary_particles' in metrics


def test_analyze_empty(engine):
    """Test on empty mask returns empty dict."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    img = np.zeros((50, 50), dtype=np.float32)
    metrics = engine.analyze(img, mask)
    assert metrics == {}
