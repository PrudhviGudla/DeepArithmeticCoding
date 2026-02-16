"""GRU-based neural network model for compression."""

import tensorflow as tf


def build_model(vocab_size, embedding_dim, rnn_units, batch_size, is_training=True):
    """
    Build a stateful GRU model for character-level prediction.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        rnn_units: Number of GRU units
        batch_size: Batch size (determines stateful behavior)
        is_training: If True, model is non-stateful for training
        
    Returns:
        Keras Sequential model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=(batch_size, None)),
        tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
            stateful=not is_training,
            dropout=0.2,
            recurrent_initializer='glorot_uniform'
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model



