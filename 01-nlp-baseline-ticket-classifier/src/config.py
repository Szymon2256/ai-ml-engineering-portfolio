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

# Prepare paths to save models and results
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "baseline_pipeline.joblib")

# Reports and artifacts
REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
REPORT_PATH = os.path.join(REPORT_DIR, "baseline_report.md")
CM_PLOT_PATH = os.path.join(FIGURES_DIR, "confusion_matrix.png")

# Make sure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)