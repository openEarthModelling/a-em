#!/usr/bin/env python3
"""ATEM Analyzer - Batch Processing CLI."""
import os
import sys
import argparse
from pathlib import Path

# Ensure the library can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from atem_analyzer import BatchProcessor, PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="ATEM Analyzer - Batch EM Image Processing"
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Input directory containing EM images'
    )
    parser.add_argument(
        '--output', '-o', default='data/processed',
        help='Output directory for results'
    )
    parser.add_argument(
        '--microscope', choices=['TEM', 'SEM', 'auto'],
        default='auto',
        help='Microscope type (auto-detect from metadata if available)'
    )
    parser.add_argument(
        '--particle', default='soot',
        help='Particle type: soot, spherical, fiber, ...'
    )
    parser.add_argument(
        '--pattern', default='*.dm4',
        help='File glob pattern (default: *.dm4)'
    )
    parser.add_argument(
        '--min-area', type=int, default=100,
        help='Minimum aerosol area in pixels'
    )
    parser.add_argument(
        '--no-grabcut', action='store_true',
        help='Disable GrabCut refinement'
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = args.input if os.path.isabs(args.input) else os.path.join(project_root, args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(project_root, args.output)

    # Build config
    config = PipelineConfig(
        microscope_type=args.microscope,
        particle_type=args.particle,
        min_area=args.min_area,
        use_grabcut_refinement=not args.no_grabcut,
        output_dir=output_path,
    )

    print(f"[*] ATEM Analyzer v0.2.0")
    print(f"[*] Input:  {input_path}")
    print(f"[*] Output: {output_path}")
    print(f"[*] Config: {config.microscope_type} | {config.particle_type} | {config.segmentation_backend}")

    # Run batch processing
    processor = BatchProcessor(config)
    df = processor.process_directory(input_path, output_path, args.pattern)

    # Save summary
    if not df.empty:
        summary_path = Path(output_path) / 'batch_summary.csv'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_path, index=False)
        print(f"[*] Summary saved: {summary_path}")
        print(f"[*] Total aerosols analyzed: {len(df)}")
    else:
        print("[!] No aerosols detected in any file.")


if __name__ == '__main__':
    main()
