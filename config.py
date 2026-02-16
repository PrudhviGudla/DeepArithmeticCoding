"""Configuration and hyperparameters for Deep Arithmetic Coding."""

# Dataset paths
TEMPLATE_FILE = "./data/llm_raw_templates.txt"
TRAINING_DATASET = "./data/training_data_hybrid.txt"
VALIDATION_DATASET = "./data/validation_data_hybrid.txt"
TESTING_DATASET = "./data/testing_data_hybrid.txt"

# Model hyperparameters
BATCH_SIZE = 64
EMBEDDING_DIM = 16
RNN_UNITS = 128
EPOCHS = 20
LEARNING_RATE = 0.0001

# Training
CHECKPOINT_DIR = "./training_checkpoints"
BEST_MODEL_PATH = "best_model.keras"
COMPRESSOR_MODEL_PATH = "gru_compressor.keras"
VOCAB_PATH = "vocab.pkl"

# Arithmetic Coder
AC_PRECISION = 32

# Base vocabulary
BASE_VOCAB = r"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ "

# Data generation
RANDOM_SEED = 42
HYBRID_RATIO = 0.4  # Ratio of machine data (40% machine, 60% natural language)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
TRAIN_LINES = 50000
VAL_LINES = 2000
TEST_LINES = 2000

# Bucket boundaries for variable-length sequence batching
# Set automatically based on data analysis, or override here
BUCKET_BOUNDARIES = [25, 45, 65]  # Will be suggested by prepare_data.py
