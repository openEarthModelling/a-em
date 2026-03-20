import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AerosolReporter:
    """
    Handles result visualization, CSV export, and summary reporting.
    """
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_labeled_mask(self, base_name, labels, num_labels):
        """
        Saves a color-coded mask of all detected aerosols.
        """
        aerosol_labeled_mask = (labels * (255 // (num_labels if num_labels > 0 else 1))).astype(np.uint8)
        path = os.path.join(self.output_dir, f"{base_name}_all_aerosols_labeled.png")
        cv2.imwrite(path, aerosol_labeled_mask)
        return path

    def save_aerosol_results(self, base_name, aerosol_obj, pix2nm):
        """
        Saves individual aerosol mask, tagged image, and analysis plots.
        """
        a_id = aerosol_obj.aerosol_id
        metrics = aerosol_obj.metrics
        
        # 1. Mask
        mask_path = os.path.join(self.output_dir, f"{base_name}_aerosol_{a_id}_mask.png")
        cv2.imwrite(mask_path, aerosol_obj.mask_roi)
        
        # 2. Tagged Image (Draw on ROI or full image?)
        # For simplicity, we'll draw on the provided image_roi
        img_out = cv2.cvtColor(aerosol_obj.image_roi, cv2.COLOR_GRAY2BGR)
        if 'primary_particles' in metrics:
            for p in metrics['primary_particles']:
                cv2.circle(img_out, p['center'], int(p['radius']), (0, 255, 0), 1)
                cv2.circle(img_out, p['center'], 1, (0, 0, 255), -1)
        
        tagged_path = os.path.join(self.output_dir, f"{base_name}_aerosol_{a_id}_tagged.png")
        cv2.imwrite(tagged_path, img_out)
        
        # 3. Df Plot
        if 'log_data' in metrics:
            log_rg_r0, log_n = metrics['log_data']
            plt.figure(figsize=(8, 6))
            plt.scatter(log_rg_r0, log_n, c='red', s=10, label='Data points')
            # N = k * (Rg/R0)^Df  => log(N) = Df * log(Rg/R0) + log(k)
            # We use the fit slope (Df) and pass through the last point for visualization
            plt.plot(log_rg_r0, log_rg_r0 * metrics['df'] + (log_n[-1] - log_rg_r0[-1] * metrics['df']), 
                     'b-', label=f'Fit (Df={metrics["df"]:.2f})')
            plt.xlabel('log(Rg/R0)')
            plt.ylabel('log(N)')
            plt.title(f'Fractal Dimension - Aerosol {a_id}')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, f"{base_name}_aerosol_{a_id}_df_plot.png"))
            plt.close()

    def export_summary(self, base_name, summary_data, pix2nm):
        """
        Exports all aerosol metrics to a CSV file.
        """
        if not summary_data:
            return None
            
        formatted_data = []
        for item in summary_data:
            m = item['metrics']
            formatted_data.append({
                "Aerosol_ID": item['id'],
                "Primary_Particles": m.get('num_particles', 0),
                "Df": m.get('df', 0),
                "Rg_nm": m.get('rg', 0) * pix2nm,
                "R0_nm": m.get('r0', 0) * pix2nm,
                "Feret_Dia_nm": m.get('feret_dia', 0) * pix2nm,
                "Convexity": m.get('convexity', 0),
                "Roundness": m.get('roundness', 0)
            })
            
        df = pd.DataFrame(formatted_data)
        path = os.path.join(self.output_dir, f"{base_name}_summary.csv")
        df.to_csv(path, index=False, encoding="UTF-8")
        return path
