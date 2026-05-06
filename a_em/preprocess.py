import cv2
import numpy as np

class AerosolPreprocessor:
    """
    Standardizes TEM image quality through contrast enhancement and noise reduction.
    """
    @staticmethod
    def process(img, filter_type='gaussian', **kwargs):
        """
        Main preprocessing workflow.
        
        Args:
            img: Input image
            filter_type: 'gaussian' or 'bilateral'
            **kwargs: Parameters for CLAHE, morphology, and filtering
        """
        clahe_clip = kwargs.get('clahe_clip', 3.0)
        clahe_tile = kwargs.get('clahe_tile', (8, 8))
        morph_kernel = kwargs.get('morph_kernel', 25)
        
        enhanced = AerosolPreprocessor.apply_clahe(img, clahe_clip, clahe_tile)
        blackhat = AerosolPreprocessor.remove_background(enhanced, morph_kernel)
        
        if filter_type == 'bilateral':
            d = kwargs.get('d', 9)
            sc = kwargs.get('sigma_color', 75)
            ss = kwargs.get('sigma_space', 75)
            denoised = AerosolPreprocessor.apply_bilateral_filter(blackhat, d, sc, ss)
        else:
            k = kwargs.get('k', 3)
            denoised = AerosolPreprocessor.denoise(blackhat, k)
            
        return denoised

    @staticmethod
    def apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

    @staticmethod
    def remove_background(img, kernel_size=25):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    @staticmethod
    def denoise(img, k=3):
        return cv2.GaussianBlur(img, (k, k), 0)

    @staticmethod
    def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
        """
        Edge-preserving filter that smooths noise while maintaining sharp boundaries.
        """
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
