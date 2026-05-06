"""Image I/O using HyperSpy for EM format support and automatic calibration."""
import os

import hyperspy.api as hs
import numpy as np


class HyperSpyReader:
    """Handles reading EM images via HyperSpy, detecting scales from metadata."""

    @staticmethod
    def load(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        signal = hs.load(path)
        if isinstance(signal, list):
            signal = signal[0]
        if not isinstance(signal, hs.signals.Signal2D):
            raise ValueError(f"Expected 2D signal, got {type(signal)}")
        if not signal.metadata.General.title:
            signal.metadata.General.title = os.path.splitext(os.path.basename(path))[0]
        return signal

    @staticmethod
    def get_scale_nm(signal) -> float:
        scale = signal.axes_manager[0].scale
        units = signal.axes_manager[0].units
        if units in ('um', 'µm', 'micrometer'):
            return scale * 1000.0
        elif units in ('nm', 'nanometer'):
            return scale
        elif units in ('mm',):
            return scale * 1e6
        elif units in ('pm',):
            return scale * 1e-3
        else:
            raise ValueError(f"Unknown units for scale conversion: '{units}'")

    @staticmethod
    def get_microscope_info(signal) -> dict:
        info = {'type': 'TEM', 'beam_energy': None, 'magnification': None}
        md = signal.metadata
        if 'Acquisition_instrument' not in md:
            return info
        inst = md.Acquisition_instrument
        if 'TEM' in inst:
            info['type'] = 'TEM'
            tem = inst.TEM
            info['beam_energy'] = (
                getattr(tem, 'beam_energy', None)
                if hasattr(tem, 'beam_energy') else None
            )
            info['magnification'] = (
                getattr(tem, 'magnification', None)
                if hasattr(tem, 'magnification') else None
            )
        elif 'SEM' in inst:
            info['type'] = 'SEM'
            sem = inst.SEM
            info['beam_energy'] = (
                getattr(sem, 'beam_energy', None)
                if hasattr(sem, 'beam_energy') else None
            )
            info['magnification'] = (
                getattr(sem, 'magnification', None)
                if hasattr(sem, 'magnification') else None
            )
        return info

    @staticmethod
    def to_uint8(signal):
        data = signal.data
        if data.dtype == np.uint8:
            return signal.deepcopy()
        dmin, dmax = data.min(), data.max()
        if dmax == dmin:
            normalized = np.zeros_like(data, dtype=np.uint8)
        else:
            normalized = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
        result = signal.deepcopy()
        result.data = normalized
        return result
