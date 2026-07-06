import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, 'tickets_dataset.csv')
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
LANGUAGES_PATH = os.path.join(RAW_DATA_DIR, 'languages.json')
VOCAB_PATH = os.path.join(PROCESSED_DATA_DIR, 'vocab.json')
LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, 'label_map.json')
REVERSE_LABEL_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, 'reverse_label_map.json')

# Prepare paths to save models and results
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Reports and artifacts
REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
REPORT_PATH = os.path.join(REPORT_DIR, "pytorch_report.md")
CM_PLOT_PATH = os.path.join(FIGURES_DIR, "confusion_matrix_embeddings.png")

# Make sure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

#---------------------------------------------------------
# Hyperparameters and settings
#---------------------------------------------------------

# Training settings
TEXT_COL = "cleaned_context"
TARGET_COL = "queue"

# Model hyperparameters
MAX_SEQUENCE_LENGTH = 256
VOCAB_SIZE = 10000
BATCH_SIZE = 32
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1