import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import src.config as config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model(training_col: str, target_col: str):
    logger.info("Loading the test data...")
    test_df = pd.read_csv(config.TEST_DATA_PATH)

    test_df[training_col] = test_df[training_col].fillna("")

    X_test = test_df[training_col]
    y_test = test_df[target_col]

    logger.info(f"Loading the trained model from {config.MODEL_PATH}...")
    pipeline = joblib.load(config.MODEL_PATH)

    logger.info("Model loaded successfully. Making predictions on the test set...")
    y_pred = pipeline.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred)
    cm = plot_confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    return metrics, cm


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics for the model predictions."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted")
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels, save_plot = False):
    """Plot the confusion matrix for the model predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_plot:
        plt.savefig(config.CM_PLOT_PATH)
        logger.info(f"Confusion matrix plot saved to {config.CM_PLOT_PATH}")
    else:
        plt.show()
    plt.close()

    return cm

