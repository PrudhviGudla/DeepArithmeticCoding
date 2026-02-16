"""
Dataset generation and analysis script.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import *
from src.utils import set_seeds, analyze_line_lengths
from src.data_generation import fetch_templates_from_gemini, prepare_datasets


def suggest_bucket_boundaries(file_path, percentiles=[25, 50, 75, 90, 95]):
    """
    Analyze line lengths and suggest bucket boundaries.
    
    Args:
        file_path: Path to dataset file
        percentiles: List of percentiles to calculate
        
    Returns:
        Suggested bucket boundaries
    """
    print(f"\n--- Suggesting Bucket Boundaries ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_lengths = [len(line.strip()) for line in f if line.strip()]
        
        if not line_lengths:
            print(f"No lines found in {file_path}")
            return None
        
        print(f"\nLine Length Statistics:")
        print(f"  Min: {np.min(line_lengths)}")
        print(f"  Max: {np.max(line_lengths)}")
        print(f"  Mean: {np.mean(line_lengths):.1f}")
        print(f"  Median: {np.median(line_lengths):.0f}")
        print(f"  Std Dev: {np.std(line_lengths):.2f}")
        
        print(f"\nPercentiles:")
        percentile_values = []
        for p in percentiles:
            val = np.percentile(line_lengths, p)
            print(f"  {p}th percentile: {val:.0f}")
            percentile_values.append(int(val))
        
        # Suggest boundaries at 33, 67 percentiles (dividing into 3 buckets)
        boundaries = [
            int(np.percentile(line_lengths, 33)),
            int(np.percentile(line_lengths, 67))
        ]
        
        print(f"\nSuggested BUCKET_BOUNDARIES: {boundaries}")
        print(f"Update config.py with: BUCKET_BOUNDARIES = {boundaries}")
        
        return boundaries
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None


def plot_length_distribution(file_path, dataset_name):
    """Plot and save line length distribution."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_lengths = [len(line.strip()) for line in f if line.strip()]
        
        if not line_lengths:
            return
        
        plt.figure(figsize=(10, 5))
        sns.histplot(line_lengths, bins=50, kde=True, color='skyblue')
        plt.title(f'Line Length Distribution - {dataset_name}')
        plt.xlabel('Line Length (characters)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        
        output_file = f"length_distribution_{dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Saved {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error plotting {dataset_name}: {e}")


def main():
    """Main data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate and analyze IoT sensor datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate default datasets:
    python scripts/prepare_data.py
    
  Generate with custom sizes:
    python scripts/prepare_data.py --train-lines 100000 --val-lines 5000 --test-lines 5000
    
  Change hybrid ratio (60% machine, 40% natural language):
    python scripts/prepare_data.py --hybrid-ratio 0.6
    
  Skip length analysis:
    python scripts/prepare_data.py --skip-analysis
        """
    )
    
    parser.add_argument(
        '--train-lines',
        type=int,
        default=TRAIN_LINES,
        help=f'Number of training examples (default: {TRAIN_LINES:,})'
    )
    parser.add_argument(
        '--val-lines',
        type=int,
        default=VAL_LINES,
        help=f'Number of validation examples (default: {VAL_LINES:,})'
    )
    parser.add_argument(
        '--test-lines',
        type=int,
        default=TEST_LINES,
        help=f'Number of test examples (default: {TEST_LINES:,})'
    )
    parser.add_argument(
        '--hybrid-ratio',
        type=float,
        default=HYBRID_RATIO,
        help=f'Ratio of machine data (0-1, default: {HYBRID_RATIO}). 0.4 = 40% machine, 60% natural.'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=TRAIN_SPLIT,
        help=f'Fraction of templates for training (default: {TRAIN_SPLIT})'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=VAL_SPLIT,
        help=f'Fraction of templates for validation (default: {VAL_SPLIT})'
    )
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip length analysis and bucket suggestion'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Gemini API key (or use GEMINI_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.hybrid_ratio < 0 or args.hybrid_ratio > 1:
        print("Error: --hybrid-ratio must be between 0 and 1")
        return
    
    if args.train_split + args.val_split > 1:
        print("Error: --train-split + --val-split must be <= 1")
        return
    
    set_seeds(RANDOM_SEED)
    
    # Load .env file for secrets
    load_dotenv()
    
    # Get API key for template generation (priority: CLI arg > .env > environment variable)
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Create .env file or set --api-key / GEMINI_API_KEY env var")
    
    # Generate templates (if needed)
    if api_key and not os.path.exists(TEMPLATE_FILE):
        fetch_templates_from_gemini(TEMPLATE_FILE, api_key)
    
    # Prepare datasets
    print("=" * 60)
    print("DATASET GENERATION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Training lines: {args.train_lines:,}")
    print(f"  Validation lines: {args.val_lines:,}")
    print(f"  Test lines: {args.test_lines:,}")
    print(f"  Hybrid ratio: {args.hybrid_ratio:.1%} machine, {(1-args.hybrid_ratio):.1%} natural language")
    print(f"  Template split: {args.train_split:.0%} train, {args.val_split:.0%} val, {(1-args.train_split-args.val_split):.0%} test")
    print("=" * 60)
    
    prepare_datasets(
        TEMPLATE_FILE,
        TRAINING_DATASET,
        VALIDATION_DATASET,
        TESTING_DATASET,
        train_split=args.train_split,
        val_split=args.val_split,
        train_lines=args.train_lines,
        val_lines=args.val_lines,
        test_lines=args.test_lines,
        hybrid_ratio=args.hybrid_ratio
    )
    
    # Analyze datasets
    if not args.skip_analysis:
        print("\n" + "=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)
        
        analyze_line_lengths(TRAINING_DATASET, "Training Dataset")
        analyze_line_lengths(VALIDATION_DATASET, "Validation Dataset")
        analyze_line_lengths(TESTING_DATASET, "Testing Dataset")
        
        # Suggest buckets based on training data
        print("\n" + "=" * 60)
        print("BUCKET BOUNDARY SUGGESTION")
        print("=" * 60)
        boundaries = suggest_bucket_boundaries(TRAINING_DATASET)
        
        if boundaries:
            print(f"\nUpdate config.py:")
            print(f"  BUCKET_BOUNDARIES = {boundaries}")
        
        # Plot distributions
        print("\n" + "=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        plot_length_distribution(TRAINING_DATASET, "Training Dataset")
        plot_length_distribution(VALIDATION_DATASET, "Validation Dataset")
        plot_length_distribution(TESTING_DATASET, "Testing Dataset")
    
    print("Data preparation complete!")
    
    print("\nNext steps:")
    print("  1. Review the length distributions and update BUCKET_BOUNDARIES in config.py")
    print("  2. Run: python scripts/train.py")


if __name__ == "__main__":
    main()
