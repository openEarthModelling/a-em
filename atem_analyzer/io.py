import os
import cv2
import numpy as np
import pandas as pd

class AerosolReader:
    """
    Handles reading TEM images, detecting scales, and managing metadata.
    """
    @staticmethod
    def parse_filename(filepath):
        """
        Extracts scale values and types from filenames (e.g., Tv94-200-fx.tif -> 200, 'fx').
        """
        filename = os.path.basename(filepath)
        name_parts = os.path.splitext(filename)[0].split("-")
        
        scale_val = 200  # Default 200nm
        bar_type = "bd"  # Default newer scale type
        
        if len(name_parts) >= 2:
            for part in name_parts[1:]:
                if part.isdigit():
                    scale_val = int(part)
                elif part.lower() in ["bd", "fx"]:
                    bar_type = part.lower()
                    
        return scale_val, bar_type

    @staticmethod
    def get_pix_to_nm(img, scale_val, bar_type):
        """
        Calculates the conversion ratio (nm per pixel) by detecting the scale bar.
        """
        bar_len_pix = AerosolReader._detect_bar_length(img, bar_type)
        if bar_len_pix == 0:
            return 1.0  # Fallback
        return scale_val / bar_len_pix

    @staticmethod
    def _detect_bar_length(img, bar_type):
        h, w = img.shape
        length = 0
        if bar_type == 'fx':
            bar_area = img[int(0.97*h):h, int(0.52*w):w]
            _, bar_div = cv2.threshold(bar_area, 127, 255, cv2.THRESH_BINARY)
            for i in range(1, bar_div.shape[1]-1):
                if np.sum(bar_div[:,i]) <= (bar_div.shape[0]-6)*255:
                    if np.sum(bar_div[:,i-1]) == np.sum(bar_div[:,i])+7*255:
                        length = i
                    if np.sum(bar_div[:,i+1]) == np.sum(bar_div[:,i])+7*255:
                        length = i - length
                        break
        elif bar_type == 'bd':
            bar_area = img[int(0.85*h):int(0.95*h), 0:int(0.3*w)]
            _, bar_div = cv2.threshold(bar_area, 20, 255, cv2.THRESH_BINARY)
            row_sums = np.sum(bar_div, axis=1)
            mid_y = np.argmin(row_sums)
            bar_list = []
            current_len = 0
            for p in range(1, bar_area.shape[1]):
                if bar_div[mid_y, p] == 0:
                    current_len += 1
                else:
                    if current_len > 0:
                        bar_list.append(current_len)
                    current_len = 0
            length = max(bar_list) if bar_list else 0
        return length

    @staticmethod
    def read_image(path, grayscale=True):
        if grayscale:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.imread(path)
