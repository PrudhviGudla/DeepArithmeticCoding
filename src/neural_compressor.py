"""Neural compression using GRU + Arithmetic Coding."""

import os
import pickle
import numpy as np
import tensorflow as tf

class SimpleArithmeticCoder:
    """arithmetic encoder/decoder with label smoothing."""
    
    def __init__(self, precision=32):
        """Initialize the coder with a given precision."""
        self.MAX_VAL = (1 << precision) - 1
        self.QUARTER = 1 << (precision - 2)
        self.HALF = 1 << (precision - 1)
        self.THREE_QUARTER = self.QUARTER * 3

    def get_freqs_smoothed(self, probs):
        """Convert probabilities to cumulative frequencies with label smoothing."""
        SCALE = 1 << 16
        freqs = (probs * SCALE).astype(np.int64)
        freqs += 1  # Label smoothing
        cum_freqs = np.cumsum(freqs)
        cum_freqs = np.insert(cum_freqs, 0, 0)
        return cum_freqs.tolist(), int(cum_freqs[-1])

    def write_bit(self, bit, pending, f, buff, count):
        """Write a single bit to file with pending bit accumulation."""
        def out(b):
            nonlocal buff, count
            buff = (buff << 1) | b
            count += 1
            if count == 8:
                f.write(bytes([buff]))
                buff = 0
                count = 0
        
        out(bit)
        for _ in range(pending):
            out(1 - bit)
        return buff, count


class NeuralCompressor:
    """Character-level neural compressor using GRU predictions + Arithmetic Coding."""
    
    def __init__(self, model_path, vocab_path, precision=32):
        """
        Initialize the compressor.
        
        Args:
            model_path: Path to saved Keras model
            vocab_path: Path to vocab pickle file
            precision: Arithmetic coder precision (bits)
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        
        if not self.model.layers[1].stateful:
            print("WARNING: Model is NOT stateful. Compression quality may be poor.")
        else:
            print("Model loaded. Stateful RNN enabled.")
        
        self.precision = precision
        with open(vocab_path, 'rb') as f:
            d = pickle.load(f)
            self.char2idx = d['char2idx']
            self.idx2char = d['idx2char']
        
        if '\n' not in self.char2idx:
            raise ValueError("Newline character not in vocabulary")
        
        self.eos_idx = self.char2idx['\n']
        self.vocab_size = len(self.char2idx)
        self.coder = SimpleArithmeticCoder(precision=precision)

    def _predict_probs(self, idx):
        """Get probability distribution for next character."""
        p = self.model(tf.expand_dims([idx], 0))
        return tf.nn.softmax(tf.squeeze(p, 0)).numpy()[0]

    def compress(self, text, out_path):
        """
        Compress text using arithmetic coding with GRU predictions.
        
        Args:
            text: Input text string
            out_path: Output file path
        """
        # Reset RNN state
        for layer in self.model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
        
        text = text + '\n'
        indices = [self.char2idx.get(c, 0) for c in text]
        low, high = 0, self.coder.MAX_VAL
        pending, buff, count = 0, 0, 0
        prev = 0
        
        with open(out_path, 'wb') as f:
            for i, idx in enumerate(indices):
                probs = (np.ones(self.vocab_size) / self.vocab_size 
                        if i == 0 else self._predict_probs(prev))
                cum, tot = self.coder.get_freqs_smoothed(probs)
                
                rng = high - low + 1
                new_high = low + (rng * int(cum[idx+1])) // tot - 1
                new_low = low + (rng * int(cum[idx])) // tot
                high, low = new_high, new_low
                
                while True:
                    if high < self.coder.HALF:
                        buff, count = self.coder.write_bit(0, pending, f, buff, count)
                        pending = 0
                        low, high = low << 1, (high << 1) | 1
                    elif low >= self.coder.HALF:
                        buff, count = self.coder.write_bit(1, pending, f, buff, count)
                        pending = 0
                        low = (low - self.coder.HALF) << 1
                        high = ((high - self.coder.HALF) << 1) | 1
                    elif low >= self.coder.QUARTER and high < self.coder.THREE_QUARTER:
                        pending += 1
                        low = (low - self.coder.QUARTER) << 1
                        high = ((high - self.coder.QUARTER) << 1) | 1
                    else:
                        break
                prev = idx
            
            pending += 1
            bit = 0 if low < self.coder.QUARTER else 1
            buff, count = self.coder.write_bit(bit, pending, f, buff, count)
            if count > 0:
                f.write(bytes([buff << (8 - count)]))

    def decompress(self, in_path):
        """
        Decompress text from arithmetic-coded file.
        
        Args:
            in_path: Input compressed file path
            
        Returns:
            Decompressed text string
        """
        # Reset RNN state
        for layer in self.model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
        
        with open(in_path, 'rb') as f:
            b = f.read()
        
        def bits():
            for byte in b:
                for i in range(7, -1, -1):
                    yield (byte >> i) & 1
            while True:
                yield 0
        
        gen = bits()
        val = 0
        for _ in range(self.precision):
            val = (val << 1) | next(gen)
        
        low, high = 0, self.coder.MAX_VAL
        res, prev = [], 0
        
        while True:
            probs = (np.ones(self.vocab_size) / self.vocab_size 
                    if len(res) == 0 else self._predict_probs(prev))
            
            cum, tot = self.coder.get_freqs_smoothed(probs)
            
            rng = high - low + 1
            s_val = ((val - low + 1) * tot - 1) // rng
            
            sym = 0
            for k in range(len(cum) - 1):
                if cum[k + 1] > s_val:
                    sym = k
                    break
            
            if sym == self.eos_idx:
                break
            
            new_high = low + (rng * int(cum[sym+1])) // tot - 1
            new_low = low + (rng * int(cum[sym])) // tot
            high, low = new_high, new_low
            
            while True:
                if high < self.coder.HALF:
                    pass
                elif low >= self.coder.HALF:
                    val = val - self.coder.HALF
                    low = low - self.coder.HALF
                    high = high - self.coder.HALF
                elif low >= self.coder.QUARTER and high < self.coder.THREE_QUARTER:
                    val = val - self.coder.QUARTER
                    low = low - self.coder.QUARTER
                    high = high - self.coder.QUARTER
                else:
                    break
                
                low = low << 1
                high = (high << 1) | 1
                val = (val << 1) | next(gen)
            
            res.append(self.idx2char[sym])
            prev = sym
        
        return "".join(res)
