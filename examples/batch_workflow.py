"""Batch processing example for a directory of DM4 files."""
from atem_analyzer import BatchProcessor, PipelineConfig

# Configure the pipeline
config = PipelineConfig(
    microscope_type='TEM',
    particle_type='soot',
    min_area=100,
    output_dir='data/processed',
)

# Process all DM4 files in the directory
processor = BatchProcessor(config)
df = processor.process_directory(
    '/data/zhangfan_data/pangyuner/H4',
    output_dir='data/processed',
    pattern='*.dm4',
)

print(f"Total aerosols analyzed: {len(df)}")
if not df.empty:
    df.to_csv('data/processed/batch_summary.csv', index=False)
    print(df.head())
