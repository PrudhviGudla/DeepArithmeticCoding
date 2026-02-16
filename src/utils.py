"""Utility functions for data processing and analysis."""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def set_seeds(seed_value):
    """Set random seeds for reproducibility across all libraries."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def analyze_line_lengths(file_path, dataset_name):
    """Analyze and plot the distribution of line lengths in a dataset."""
    print(f"\n--- Analyzing {dataset_name} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_lengths = [len(line.strip()) for line in f if line.strip()]

        if not line_lengths:
            print(f"No lines found in {file_path}.")
            return

        # Calculate summary statistics
        min_len = np.min(line_lengths)
        max_len = np.max(line_lengths)
        mean_len = np.mean(line_lengths)
        median_len = np.median(line_lengths)
        std_dev_len = np.std(line_lengths)

        print(f"Number of lines: {len(line_lengths)}")
        print(f"Min length: {min_len:.0f} chars")
        print(f"Max length: {max_len:.0f} chars")
        print(f"Mean length: {mean_len:.2f} chars")
        print(f"Median length: {median_len:.0f} chars")
        print(f"Std Dev length: {std_dev_len:.2f} chars")

        # Plotting the distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(line_lengths, bins=50, kde=True, color='skyblue')
        plt.title(f'Distribution of Line Lengths in {dataset_name}')
        plt.xlabel('Line Length (characters)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")


def plot_training_history(history):
    """Plot training and validation accuracy/loss curves."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))

    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    # Plot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
