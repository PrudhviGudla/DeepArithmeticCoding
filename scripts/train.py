"""Training script for GRU compression model."""

import os
import sys
import pickle
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import *
from src.utils import set_seeds, analyze_line_lengths, plot_training_history
from src.model import build_model
from src.data_generation import prepare_datasets

def loss_fn(labels, logits):
    """Sparse categorical crossentropy loss."""
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def create_dataset(data_file, mode="train"):
    """Create bucketed dataset from text file."""
    print(f"Creating {mode} dataset\n")
    print("Loading text from file...")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Could not find {data_file}")
        exit(1)
    
    # Build vocabulary
    unique_chars = sorted(set(BASE_VOCAB) | set("".join(lines)) | {'\n'})
    char2idx = {u: i+1 for i, u in enumerate(unique_chars)}
    idx2char = ['<PAD>'] + unique_chars
    vocab_size = len(idx2char)
    
    print(f"Loaded {len(lines)} lines. Vocab Size: {vocab_size}")
    
    # Save vocabulary (overwrite each call for consistency)
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump({'char2idx': char2idx, 'idx2char': idx2char}, f)
    
    def line_generator():
        for line in lines:
            text = line + '\n'
            encoded = [char2idx.get(c, 0) for c in text]
            yield encoded[:-1], encoded[1:]
    
    dataset = tf.data.Dataset.from_generator(
        line_generator,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    
    bucket_batch_sizes = [BATCH_SIZE] * (len(BUCKET_BOUNDARIES) + 1)
    dataset = dataset.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x)[0],
        bucket_boundaries=BUCKET_BOUNDARIES,
        bucket_batch_sizes=bucket_batch_sizes,
        pad_to_bucket_boundary=False
    )
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    print("Bucketed Dataset created\n")
    
    return dataset, vocab_size


def main():
    """Main training pipeline."""
    set_seeds(RANDOM_SEED)
    
    # Prepare datasets (using config parameters)
    prepare_datasets(
        TEMPLATE_FILE,
        TRAINING_DATASET,
        VALIDATION_DATASET,
        TESTING_DATASET,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        train_lines=TRAIN_LINES,
        val_lines=VAL_LINES,
        test_lines=TEST_LINES,
        hybrid_ratio=HYBRID_RATIO
    )
    
    # Analyze data
    # analyze_line_lengths(TRAINING_DATASET, "Training Dataset")
    # analyze_line_lengths(VALIDATION_DATASET, "Validation Dataset")
    # analyze_line_lengths(TESTING_DATASET, "Testing Dataset")
    
    # Create datasets
    train_ds, vocab_size = create_dataset(TRAINING_DATASET, "train")
    val_ds, _ = create_dataset(VALIDATION_DATASET, "val")
    test_ds, _ = create_dataset(TESTING_DATASET, "test")
    
    # Check GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"GPU Detected: {physical_devices[0]}\n")
    else:
        print("No GPU found\n")
    
    # Build model
    model = build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE, is_training=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # Callbacks
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train
    print(f"Starting Training for {EPOCHS} epochs...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Plot results
    plot_training_history(history)
    
    # Evaluate
    print("\n--- Eval on Test Set ---")
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    bpc = loss / (np.log(2))
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Bits Per Character (BPC): {bpc:.4f}\n")
    
    # Export inference model
    print("Exporting Model for Compression...")
    
    model_pred = build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, batch_size=1, is_training=False)
    model_pred.load_weights(BEST_MODEL_PATH)
    model_pred.build(tf.TensorShape([1, None]))
    model_pred.save(COMPRESSOR_MODEL_PATH)
    print(f"Done! Saved {COMPRESSOR_MODEL_PATH}")


if __name__ == "__main__":
    main()
