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

    def __repr__(self):
        return f"<AerosolObject ID={self.aerosol_id} metrics={list(self.metrics.keys())}>"
