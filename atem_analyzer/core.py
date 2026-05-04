import numpy as np

class AerosolObject:
    """
    Data model representing a single aerosol particle extracted from a TEM image.
    Stores the Region of Interest (ROI) image, its corresponding mask, and calculated metrics.
    """
    def __init__(self, aerosol_id, image_roi, mask_roi, offset=(0, 0), bbox=None, area=None):
        self.aerosol_id = aerosol_id
        self.image_roi = image_roi
        self.mask_roi = mask_roi
        self.offset = offset  # (y, x) top-left corner in the original image
        self.bbox = bbox      # (x, y, w, h)
        self.area = area      # Calculated area in pixels
        self.metrics = {}

    def add_metric(self, name, value):
        self.metrics[name] = value

    def get_physical_scale_nm(self):
        """Return nm/pixel from the ROI's calibration if available."""
        try:
            import hyperspy.api as hs
            if isinstance(self.image_roi, hs.signals.Signal2D):
                scale = self.image_roi.axes_manager[0].scale
                units = self.image_roi.axes_manager[0].units
                if units in ('um', 'µm', 'micrometer'):
                    return scale * 1000.0
                elif units in ('nm', 'nanometer'):
                    return scale
                elif units in ('mm',):
                    return scale * 1e6
                else:
                    return 1.0
        except ImportError:
            pass
        return 1.0

    def __repr__(self):
        return f"<AerosolObject ID={self.aerosol_id} metrics={list(self.metrics.keys())}>"
