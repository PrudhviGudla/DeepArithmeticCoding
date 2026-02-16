# Deep Arithmetic Coding

A **lossless neural compression** framework combining a **Character-Level GRU** (Probability Engine) with **Arithmetic Coding** for compressing high-entropy IoT sensor data, URLs, error codes, and machine-generated text.

## Problem Statement

### The Bandwidth Problem in IoT

JSON-LD data and IoT/WoT sensor data contain high-entropy text and URLs that cannot be substituted using static dictionaries across devices. Traditional compression approaches fail:

- **CBOR/Standard Formats**: Require 8 bits per character (1 byte/char) + extra bytes for length encoding
- **Static Dictionaries**: Don't work across diverse deployment contexts
- **GZIP/ZIP**: Too heavy for embedded devices; designed for file compression, not streaming data

### The Trade-off: Bandwidth vs. Compute

In IoT systems, **bandwidth is expensive**:
- Sending 100 bytes over LoRaWAN/NB-IoT → **seconds of transmission** → drains radio battery
- Running a tiny CharGRU on ESP32 → **milliseconds of computation** → minimal CPU battery drain

**This project inverts the trade-off:** Use the compute power that exists to save expensive bandwidth.

## The Idea: Neural Compression with Arithmetic Coding

Instead of static compression tables, use a **deep learning model trained on domain data** to predict character probabilities. Then use **Arithmetic Encoding** to convert those predictions into the minimum number of bits.

### How It Works

1. **All devices deploy the same CharGRU model** → They generate identical probability predictions for any input
2. **Sender**: GRU predicts next-character probability → Arithmetic Coder encodes using that probability → Output: minimal bits
3. **Receiver**: Reverse process with RNN state tracking → Bit-by-bit decoding → Perfect reconstruction

### No Length Encoding Needed

Unlike CBOR, we don't need to encode the string length because:
- The vocabulary includes `\n` (end-of-line marker)
- The RNN predicts its probability like any other character
- Decoder recognizes when `\n` is decoded → stops decompression
- The entropy-encoded `\n` is part of the compressed stream

**Compression achieved: ~2-3x over CBOR, BPC reduced from 8 to 2-3 bits/char**



### Key Components

| Component | Purpose | Implementation |
|-----------|---------|-----------------|
| **Model** | Character-level GRU | Stateful, batch_size=1 for streaming prediction |
| **Vocabulary** | Char↔Index mapping | Learned from training data |
| **Arithmetic Coder** | Entropy encoding | Fixed precision with renormalization |
| **Decompressor** | Bit-level decoder | Mirrors encoder logic exactly |

## Repository Structure

```
DeepArithmeticCoding/
├── config.py                  # Hyperparameters and constants
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── src/                       # Core modules (~350 LOC)
│   ├── __init__.py            # Lazy imports (TensorFlow optional)
│   ├── utils.py               # Data analysis, plotting, seeding
│   ├── data_generation.py     # Dataset creation (Gemini API integration)
│   ├── model.py               # GRU architecture
│   └── neural_compressor.py   # Compression/decompression engine
│
├── scripts/                   # Executable entry points (with CLI args)
│   ├── prepare_data.py        # Generate data + analysis + bucket suggestions
│   ├── train.py               # Full training pipeline
│   └── compress_test.py       # Compression testing (4 modes)
│
├── Notebooks/
│   └── DAC_Development.ipynb  # Development notebook with results

```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Gemini API Key

**Option A: Use `.env` file (Recommended)**

```bash
# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your-actual-api-key-here
```

Get your API key at: https://aistudio.google.com/apikey

**Option B: Environment Variable**

```bash
# Linux/macOS
export GEMINI_API_KEY="your-gemini-api-key"

# Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-api-key"
```

**Option C: CLI Argument**

```bash
python scripts/prepare_data.py --api-key "your-gemini-api-key"
```

### 3. Generate & Analyze Dataset

```bash
python scripts/prepare_data.py
```

**This script:**
- Loads API key from `.env` (or environment variable)
- Generates 150+ IoT templates via Gemini API (cached after first run)
- Creates 50k train + 2k validation + 2k test lines
- Analyzes line-length distributions
- **Suggests optimal bucket boundaries** for variable-length batches for training and testing
- Saves visualization plots

**Review the terminal output and update `config.py`:**

```python
BUCKET_BOUNDARIES = [25, 45, 65]  # Use script suggestions to optimize RNN state partitioning
```

### 4. Train the Model

```bash
python scripts/train.py
```

**Outputs:**
- `best_model.keras` — Best model (checkpointed during training)
- `vocab.pkl` — Character vocabulary & indices
- Training/validation accuracy & loss plots
- Test set metrics (BPC, Accuracy)

### 5. Test Compression

```bash
# Test on sample strings (shows compression savings)
python scripts/compress.py --mode specific \
    --test-strings "sensor #123 reading"  "http://api.example.com"

# Batch test on dataset (empirical metrics)
python scripts/compress_test.py --mode batch --num-samples 100

# Test all modes
python scripts/compress_test.py
```


## How Arithmetic Coding Works

### The Problem: From Bytes to Bits

Traditional encoding uses 8 bits per character. Arithmetic Coding exploits probability:

- **English text**: Average char probability ~0.1 → Entropy ~3.3 bits/char
- **Machine data**: Patterns exist (URLs, numbers, errors) → Entropy ~4-5 bits/char
- **Random noise**: No patterns → Entropy ≈ 8 bits/char

### The Solution: Finite Precision Integer Arithmetic

Instead of floating-point `[0.0, 1.0]`, use fixed **32-bit integers** `[0, 2³²-1]`.

**Core Variables:**
- `Low`, `High` — Current interval boundary (integers)
- `Pending` — Bits we're unsure about (handling middle trap)
- `Value` — Decoder's rolling bit window


### Encoding Process (Compression)

**Step 1: Initialize**
```
Low = 0, High = 2³² - 1, Pending = 0
```

**Step 2: For each character:**

1. **Get Probability**: RNN predicts P(char | context)
2. **Narrow Range**: 
   ```
   Range = High - Low + 1
   High = Low + Range × CumProb(char) - 1
   Low  = Low + Range × CumProb(char-1)
   ```
3. **Renormalize** (Zoom to extract bits):
   - **Case A (Top Half)**: `Low ≥ 2³¹`
     - Output: `1` + any pending opposite bits
     - Action: Shift `Low`, `High` left (discard top 2 bits)
   - **Case B (Bottom Half)**: `High < 2³¹`
     - Output: `0` + any pending opposite bits
     - Action: Shift both left
   - **Case C (Middle Trap)**: `2³⁰ ≤ Low` and `High < 3×2³⁰` 
     - Increment `Pending` counter
     - Zoom into middle (prevent range collapse)
   - **Otherwise**: Range is wide enough, continue to next character

**Step 3: At End of Stream**

Output final bits to flush pending bits


### Decoding Process (Decompression)

**Step 1: Initialize**
- Read first 32 bits from file into `Value` register
- Set `Low = 0`, `High = 2³² - 1`

**Step 2: For each symbol:**

1. **Identify Character**: Where does `Value` fall in the cumulative probability map?
   ```
   Position = (Value - Low) / (High - Low + 1)
   Find char where CumProb(char-1) ≤ Position < CumProb(char)
   ```

2. **Narrow Range**: Same math as encoder
   ```
   High = Low + Range × CumProb(char) - 1
   Low  = Low + Range × CumProb(char-1)
   ```

3. **Synchronize Renormalization** (Read bits):
   - Decoder performs **exact same tests** as encoder
   - When encoder outputs a bit → decoder reads it and shifts `Value`
   - This keeps encoder/decoder perfectly synchronized

4. **End on `\n`**: Vocabulary contains `\n` character; decoder recognizes when it's decoded


## Results & Performance

### Laboratory Test (100 Random Lines)

Compression test on 100 randomly sampled lines from test dataset:

**Metrics:**
- **Decompression Accuracy**: 99%+ lossless reconstruction verified across all test strings
- **Average Savings**: 75-80% vs. CBOR

**Sample Results:**

| String (first 40 chars) | Original | CBOR | AC | Savings |
|-----------|----------|------|----|-----------| 
| the ground displacement speed at locati... | 56 | 62 | 15 | 75.8% |
| https://api.sensor-cloud.org/v1/dev/123... | 42 | 47 | 8 | 82.9% |
| sensor #123 temperature is 25.6 C | 34 | 39 | 6 | 84.6% |
| pressure sensor reading 1012.3 hPa at... | 54 | 60 | 13 | 78.3% |
| Lidar #4829 operating normally, 12V su... | 43 | 48 | 9 | 81.2% |

**Key Findings:**
- **URLs**: 83-85% compression — high repetition, predictable patterns
- **Sensor Descriptions**: 75-80% compression — mix of templates and variable content
- **Status Messages**: 78-82% compression — trained patterns dominate
- **Out-of-Distribution**: 40-50% compression — model less confident on unseen patterns


## Advanced Usage

### Customize Dataset Generation

```bash
# Larger dataset with custom hybrid ratio
python scripts/prepare_data.py --train-lines 100000 --hybrid-ratio 0.6

# Skip time-consuming analysis
python scripts/prepare_data.py --skip-analysis

# Custom dataset splits
python scripts/prepare_data.py --train-split 0.75 --val-split 0.15
```

### Flexible Compression Testing

```bash
# Mode 1: Single verification
python scripts/compress.py --mode single

# Mode 2: Batch metrics (100 samples)
python scripts/compress.py --mode batch --num-samples 100

# Mode 3: Custom test strings
python scripts/compress.py --mode specific \
  --test-strings "sensor #123" "http://example.com" "Error_0xAB"

# Mode 4: All tests
python scripts/compress.py --mode all --output-dir ./results
```


### Data Flow

```
Training Pipeline:
  prepare_data.py → generate dataset → analyze distribution → store to disk
  train.py → load processed data → build model → train/validate → checkpoint

Compression Pipeline:
  Text input → vocab encode → stateful GRU prediction
            → arithmetic coder → compressed bytes → output file
            
Decompression Pipeline:
  Compressed bytes → initialize decoder with RNN
            → arithmetic decode → character recovery → text output
```


## References & Inspiration

- **DeepZip (Stanford)**: RNNs beating GZIP on text compression
- **Arithmetic Coding**: Information theory optimal prefix-free codes
- **IoT Standards**: SSN/SOSA ontology for semantic sensor networks
- **WoT**: W3C Web of Things architecture


## Future Work

- [ ] **TFLite Quantization**: Export fully quantized 8-bit model for edge and Test on ESP32
- [ ] **ESP32 Deployment**: Verify inference pipeline on real hardware
- [ ] **Attention Mechanism**: Longer context for improved predictions
- [ ] **Hardware Benchmarks**: Compare with gzip, CBOR etc. on actual radio modules
- [ ] **Adaptive Models**: Per-domain specialized networks (URLs, sensor readings, logs)




