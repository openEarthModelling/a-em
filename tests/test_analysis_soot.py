import numpy as np
import pytest

from a_em.analysis.soot import SootAnalysisEngine


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


def test_analyze_with_hyperspy_like_object(engine):
    """Test analyze with an object that has a .data attribute (like HyperSpy Signal2D)."""
    cv2 = pytest.importorskip('cv2')

    mask = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(mask, (40, 40), 15, 255, -1)
    cv2.circle(mask, (55, 42), 12, 255, -1)

    img_data = np.ones((80, 80), dtype=np.float32) * 200

    # Mock a HyperSpy-like object with .data attribute
    class MockSignal:
        def __init__(self, data):
            self.data = data

    mock_signal = MockSignal(img_data)
    metrics = engine.analyze(mock_signal, mask)

    assert 'df' in metrics
    assert 'num_particles' in metrics


def test_analyze_with_boolean_mask(engine):
    """Test analyze handles boolean masks by converting to uint8."""
    cv2 = pytest.importorskip('cv2')

    mask = np.zeros((80, 80), dtype=bool)
    cv2.circle(np.uint8(mask) * 255, (40, 40), 15, 1, -1)
    mask = np.zeros((80, 80), dtype=bool)
    y, x = np.ogrid[:80, :80]
    mask[(x - 40)**2 + (y - 40)**2 <= 15**2] = True

    img = np.ones((80, 80), dtype=np.float32) * 200
    metrics = engine.analyze(img, mask)

    assert 'df' in metrics
    assert 'num_particles' in metrics
