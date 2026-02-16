"""Compression testing and benchmarking script.

Usage:
    python scripts/compress.py [--mode MODE] [--num-samples N] [--test-strings STRINGS...]
"""

import os
import sys
import random
import argparse
import numpy as np
import cbor2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import *
from src.neural_compressor import NeuralCompressor


def test_single_string(compressor, test_str):
    """Test compression of a single string."""
    print(f"\nOriginal: '{test_str}'")
    
    compressor.compress(test_str, "temp.bin")
    decompressed = compressor.decompress("temp.bin")
    
    print(f"Decoded:  '{decompressed}'")
    print(f"Match:    {decompressed == test_str}")
    
    # Compare with CBOR
    with open("temp.cbor", "wb") as f:
        cbor2.dump(test_str, f)
    
    cbor_size = os.path.getsize("temp.cbor")
    ac_size = os.path.getsize("temp.bin")
    
    print(f"CBOR Size: {cbor_size} bytes")
    print(f"AC Size:   {ac_size} bytes")
    print(f"Savings:   {(cbor_size - ac_size) / cbor_size * 100:.2f}%")
    
    # Cleanup
    os.remove("temp.bin")
    os.remove("temp.cbor")


def test_batch(compressor, data_file, num_samples=100):
    """Test compression on multiple samples."""
    print(f"\n--- Running Compression on {num_samples} Samples ---")
    
    with open(data_file, "r", encoding="utf-8") as f:
        test_lines = [line.strip() for line in f if line.strip()]
    
    lines_to_evaluate = random.sample(test_lines, min(num_samples, len(test_lines)))
    
    print(f"{'Line Content':<40} | {'Orig':<5} | {'CBOR':<5} | {'AC':<5} | {'Savings':<9} | {'Match'}")
    print("-" * 95)
    
    total_cbor = 0
    total_ac = 0
    total_chars = 0
    successful = 0
    savings_list = []
    
    for line in lines_to_evaluate:
        with open("temp.cbor", "wb") as f:
            cbor2.dump(line, f)
        cbor_len = os.path.getsize("temp.cbor")
        
        compressor.compress(line, "temp.ae")
        ac_len = os.path.getsize("temp.ae")
        
        decompressed = compressor.decompress("temp.ae")
        match = (decompressed == line)
        
        savings = (cbor_len - ac_len) / cbor_len * 100 if cbor_len > 0 else 0.0
        
        if match:
            total_cbor += cbor_len
            total_ac += ac_len
            total_chars += len(line)
            successful += 1
            savings_list.append(savings)
        
        disp_line = (line[:37] + '..') if len(line) > 37 else line
        print(f"{disp_line:<40} | {len(line):<5} | {cbor_len:<5} | {ac_len:<5} | {savings:>8.1f}% | {str(match):<5}")
    
    print("-" * 95)
    print(f"Decoding Accuracy: {successful/len(lines_to_evaluate)*100:.2f}% ({successful}/{len(lines_to_evaluate)})")
    
    if savings_list:
        print(f"Average Savings: {np.mean(savings_list):.2f}%")
    
    if total_chars > 0:
        bpc = (total_ac * 8) / total_chars
        print(f"Empirical BPC: {bpc:.2f} bits/char\n")
    
    # Cleanup
    if os.path.exists("temp.cbor"):
        os.remove("temp.cbor")
    if os.path.exists("temp.ae"):
        os.remove("temp.ae")


def main():
    """Main compression testing pipeline."""
    parser = argparse.ArgumentParser(
        description="Compress and benchmark text using neural arithmetic coding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single string test (default):
    python scripts/compress.py
    
  Batch test with 100 samples:
    python scripts/compress.py --mode batch --num-samples 100
    
  Batch test with custom dataset:
    python scripts/compress.py --mode batch --dataset ./data/custom.txt --num-samples 50
    
  Test specific strings:
    python scripts/compress.py --mode specific --test-strings "sensor #123 online" "Error_0x42_Timeout"
    
  All tests:
    python scripts/compress.py --mode all
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['single', 'batch', 'specific', 'all'],
        help='Test mode: single string, batch evaluation, specific strings, or all (default: all)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples for batch mode (default: 100)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=TESTING_DATASET,
        help=f'Dataset file for batch mode (default: {TESTING_DATASET})'
    )
    parser.add_argument(
        '--test-strings',
        type=str,
        nargs='+',
        default=[],
        help='Test strings for specific mode (space-separated)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./test',
        help='Output directory for test results (default: ./test)'
    )
    
    args = parser.parse_args()
    
    # Initialize compressors
    enc = NeuralCompressor(COMPRESSOR_MODEL_PATH, VOCAB_PATH, AC_PRECISION)
    dec = NeuralCompressor(COMPRESSOR_MODEL_PATH, VOCAB_PATH, AC_PRECISION)
    
    # Single string test
    if args.mode in ['single', 'all']:
        test_str = "the ground displacement speed at location of VCAB-DP1-BP-40"
        print("\n" + "=" * 60)
        print("SINGLE STRING TEST")
        print("=" * 60)
        test_single_string(enc, test_str)
    
    # Batch test
    if args.mode in ['batch', 'all']:
        print("\n" + "=" * 60)
        print("BATCH TEST")
        print("=" * 60)
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset not found at {args.dataset}")
            return
        test_batch(enc, args.dataset, num_samples=args.num_samples)
    
    # Specific strings test
    test_strings = args.test_strings or [
        "https://api.sensor-cloud.org/v1/dev/12345",
        "https://api.sensor-cloud.org/v1/dev/99999",
        "https://api.sensor-cloud.org/v1/dev/ABCDE",
        "Critical_System_Failure_Run_Away"
    ]
    
    if args.mode in ['specific', 'all']:
        print("\n" + "=" * 60)
        print("SPECIFIC STRINGS TEST")
        print("=" * 60)
        print(f"Testing {len(test_strings)} specific strings...\n")
        
        cbor_dir = os.path.join(args.output_dir, "cbor")
        ac_dir = os.path.join(args.output_dir, "ac")
        os.makedirs(cbor_dir, exist_ok=True)
        os.makedirs(ac_dir, exist_ok=True)
        
        print(f"{'String (first 50 chars)':<55} | {'CBOR':<6} | {'AC':<6} | {'Savings':<8} | {'Match'}")
        print("-" * 85)
        
        for i, text_string in enumerate(test_strings):
            cbor_file = os.path.join(cbor_dir, f"string_{i:02d}.cbor")
            ac_file = os.path.join(ac_dir, f"string_{i:02d}.ac")
            
            with open(cbor_file, "wb") as f:
                cbor2.dump(text_string, f)
            cbor_size = os.path.getsize(cbor_file)
            
            enc.compress(text_string, ac_file)
            ac_size = os.path.getsize(ac_file)
            
            decompressed = dec.decompress(ac_file)
            match = (decompressed == text_string)
            
            savings = (cbor_size - ac_size) / cbor_size * 100 if cbor_size > 0 else 0.0
            print(f"{text_string[:50]:<55} | {cbor_size:<6} | {ac_size:<6} | {savings:>7.2f}% | {str(match):<5}")


if __name__ == "__main__":
    main()
