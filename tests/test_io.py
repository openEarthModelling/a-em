import os
import numpy as np
import pytest
import hyperspy.api as hs

from a_em.io import HyperSpyReader

TEST_DM4 = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_dm4', '40026.dm4')


class TestHyperSpyReader:
    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            HyperSpyReader.load("/nonexistent/file.dm4")

    def test_get_scale_nm_um(self):
        s = hs.signals.Signal2D(np.zeros((100, 100)))
        s.axes_manager[0].scale = 0.5
        s.axes_manager[0].units = 'um'
        s.axes_manager[1].scale = 0.5
        s.axes_manager[1].units = 'um'
        assert HyperSpyReader.get_scale_nm(s) == 500.0

    def test_get_scale_nm_nm(self):
        s = hs.signals.Signal2D(np.zeros((100, 100)))
        s.axes_manager[0].scale = 2.5
        s.axes_manager[0].units = 'nm'
        s.axes_manager[1].scale = 2.5
        s.axes_manager[1].units = 'nm'
        assert HyperSpyReader.get_scale_nm(s) == 2.5

    def test_to_uint8_float(self):
        s = hs.signals.Signal2D(np.array([[0.0, 100.0], [200.0, 300.0]], dtype=np.float32))
        s8 = HyperSpyReader.to_uint8(s)
        assert s8.data.dtype == np.uint8
        assert s8.data[0, 0] == 0
        assert s8.data[-1, -1] == 255

    def test_to_uint8_already_uint8(self):
        s = hs.signals.Signal2D(np.array([[0, 128], [128, 255]], dtype=np.uint8))
        s8 = HyperSpyReader.to_uint8(s)
        assert s8.data.dtype == np.uint8
        np.testing.assert_array_equal(s.data, s8.data)


class TestHyperSpyReaderWithDM4:
    @pytest.fixture(scope='class')
    def dm4_signal(self):
        if not os.path.exists(TEST_DM4):
            pytest.skip(f"Test DM4 file not found: {TEST_DM4}")
        return HyperSpyReader.load(TEST_DM4)

    def test_load_dm4(self, dm4_signal):
        assert dm4_signal.data.ndim == 2
        assert dm4_signal.data.shape == (1336, 2004)

    def test_dm4_scale(self, dm4_signal):
        scale_nm = HyperSpyReader.get_scale_nm(dm4_signal)
        assert 1.0 < scale_nm < 2.0

    def test_dm4_microscope_info(self, dm4_signal):
        info = HyperSpyReader.get_microscope_info(dm4_signal)
        assert info['type'] == 'TEM'
        assert info['beam_energy'] == 200.0
