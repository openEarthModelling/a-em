"""Batch processing for directories of EM images."""
import os
from pathlib import Path

import pandas as pd

from atem_analyzer.config import PipelineConfig
from atem_analyzer.pipeline import PipelineExecutor
from atem_analyzer.io import HyperSpyReader


class BatchProcessor:
    """Processes entire directories of EM images.

    Scans a directory for matching files, optionally groups them by
    configuration, and runs the analysis pipeline on each.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config

    def process_directory(self, input_dir: str, output_dir: str = None,
                          pattern: str = '*.dm4') -> pd.DataFrame:
        """Process all matching files in a directory.

        Args:
            input_dir: Path to directory containing images.
            output_dir: Override output directory from config.
            pattern: Glob pattern for file matching.

        Returns:
            DataFrame with aggregated results from all files.
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        files = sorted(input_path.glob(pattern))
        if not files:
            print(f"[WARN] No files matching '{pattern}' in {input_dir}")
            return pd.DataFrame()

        results = []
        for filepath in files:
            print(f"[*] Processing: {filepath.name}")
            try:
                result = self._process_single_file(filepath, output_dir)
                results.append(result)
            except Exception as e:
                print(f"[!] Error processing {filepath.name}: {e}")
                results.append({'file': filepath.name, 'error': str(e)})

        return self._aggregate_results(results)

    def _process_single_file(self, filepath: Path, output_dir: str = None):
        """Process a single file and return result dict."""
        signal = HyperSpyReader.load(str(filepath))

        # Auto-configure if no config provided
        config = self.config or PipelineConfig.from_signal(signal)
        if output_dir:
            config.output_dir = output_dir

        executor = PipelineExecutor(config)
        aerosols = executor.run(signal)

        return {
            'file': filepath.name,
            'num_aerosols': len(aerosols),
            'aerosols': aerosols,
            'config': config,
        }

    def _aggregate_results(self, results: list) -> pd.DataFrame:
        """Aggregate all aerosol metrics into a single DataFrame."""
        rows = []
        for result in results:
            if 'error' in result:
                continue

            for aerosol in result['aerosols']:
                row = {
                    'file': result['file'],
                    'aerosol_id': aerosol.aerosol_id,
                }
                row.update(aerosol.metrics)

                # Convert pixel metrics to physical units
                scale_nm = aerosol.get_physical_scale_nm()
                for key in ['rg', 'r0', 'feret_dia']:
                    if key in row and scale_nm != 1.0:
                        row[f'{key}_nm'] = row[key] * scale_nm

                rows.append(row)

        return pd.DataFrame(rows)
