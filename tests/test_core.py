import numpy as np
from atem_analyzer.core import AerosolObject


def test_aerosol_object_legacy():
    img = np.zeros((100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    obj = AerosolObject(1, img, mask, offset=(10, 20), bbox=(20, 10, 100, 100), area=5000)
    assert obj.aerosol_id == 1
    assert obj.get_physical_scale_nm() == 1.0


def test_aerosol_object_add_metric():
    obj = AerosolObject(1, np.zeros((10, 10)), np.zeros((10, 10)))
    obj.add_metric('df', 1.8)
    assert obj.metrics['df'] == 1.8
