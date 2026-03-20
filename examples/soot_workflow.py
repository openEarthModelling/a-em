import os
import cv2
import numpy as np
import argparse
import sys

# Ensure the library can be found if running from the examples folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from atem_analyzer import (
    AerosolReader,
    AerosolPreprocessor,
    AerosolSegmenter,
    AerosolObject,
    AerosolReporter,
    SootAnalysisEngine
)

class ATEMAnalyzer:
    """
    Main controller for the ATEM (Aerosol Transmission Electron Microscopy) 
    image processing platform.
    """
    def __init__(self, engine_type='soot', output_dir="data/processed"):
        self.reader = AerosolReader()
        self.preprocessor = AerosolPreprocessor()
        self.segmenter = AerosolSegmenter()
        self.reporter = AerosolReporter(output_dir=output_dir)
        
        # Strategy pattern for analysis engine
        if engine_type == 'soot':
            self.engine = SootAnalysisEngine()
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

    def run_pipeline(self, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"[*] Processing: {image_path}")
        
        # 1. Load and detect scale
        img_ori = self.reader.read_image(image_path)
        if img_ori is None:
            print(f"[!] Error: Could not read {image_path}")
            return
            
        scale_val, bar_type = self.reader.parse_filename(image_path)
        pix2nm = self.reader.get_pix_to_nm(img_ori, scale_val, bar_type)
        print(f"[*] Scale: {scale_val}nm ({bar_type}), Ratio: {pix2nm:.4f} nm/pix")
        
        # 2. Preprocess (Using Bilateral Filter for better edge preservation)
        preprocessed = self.preprocessor.process(img_ori, filter_type='bilateral', d=9, sigma_color=75, sigma_space=75)
        
        # 3. Segmentation (Using Adaptive Threshold and GrabCut refinement)
        full_mask = self.segmenter.extract_mask(preprocessed, adaptive=True, refine=True)
        
        # 4. Extract all Aerosol Objects (Internal logic move to library)
        aerosols = self.segmenter.extract_all_objects(img_ori, full_mask, min_area=100)
        print(f"[*] Detected {len(aerosols)} aerosol candidates.")
        
        # Save labeled mask for visual inspection
        labeled_mask = np.zeros_like(full_mask)
        for i, obj in enumerate(aerosols):
            x, y, w, h = obj.bbox
            mask_roi_global = np.zeros_like(full_mask)
            mask_roi_global[y:y+h, x:x+w] = obj.mask_roi
            labeled_mask[mask_roi_global > 0] = i + 1
        self.reporter.save_labeled_mask(base_name, labeled_mask, len(aerosols) + 1)
        
        summary_data = []
        for a_obj in aerosols:
            print(f"[*] Analyzing Aerosol #{a_obj.aerosol_id} (Area: {a_obj.area:.1f})...")
            
            # 5. Analysis (Using Strategy)
            # Use ROI images for faster and more accurate analysis
            metrics = self.engine.analyze(a_obj.image_roi, a_obj.mask_roi)
            a_obj.metrics = metrics
            
            # 6. Reporting
            self.reporter.save_aerosol_results(base_name, a_obj, pix2nm)
            summary_data.append({"id": a_obj.aerosol_id, "metrics": metrics})
            
        # 7. Final Summary Export
        if summary_data:
            summary_path = self.reporter.export_summary(base_name, summary_data, pix2nm)
            print(f"[*] Analysis complete. Summary saved to: {summary_path}")
        else:
            print("[!] No aerosols detected.")

def main():
    parser = argparse.ArgumentParser(description="ATEM Aerosol Image Analyzer Workflow")
    # Paths are relative to the project root
    parser.add_argument("--input", default="data/raw", help="Input directory or file")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--engine", default="soot", help="Analysis engine type")
    args = parser.parse_args()
    
    # Adjust paths if they are relative
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_path = args.input if os.path.isabs(args.input) else os.path.join(project_root, args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(project_root, args.output)

    analyzer = ATEMAnalyzer(engine_type=args.engine, output_dir=output_path)
    
    # Priority logic for image.png
    target_img = os.path.join(input_path, "image.png")
    if os.path.exists(target_img):
        analyzer.run_pipeline(target_img)
    elif os.path.isdir(input_path):
        for f in os.listdir(input_path):
            if f.lower().endswith((".tif", ".png", ".jpg")):
                analyzer.run_pipeline(os.path.join(input_path, f))
    elif os.path.isfile(input_path):
        analyzer.run_pipeline(input_path)

if __name__ == "__main__":
    main()
