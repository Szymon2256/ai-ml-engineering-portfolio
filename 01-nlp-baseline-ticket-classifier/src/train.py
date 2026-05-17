import pandas as pd
import joblib
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import src.config as config
from src.evaluate import calculate_metrics, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(training_col: str, target_col: str):
    logger.info("Loading training data...")
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)

    train_df[training_col] = train_df[training_col].fillna("")

    X_train = train_df[training_col]
    y_train = train_df[target_col]

    logger.info("Initializing the model pipeline with TF-IDF and Logistic Regression...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])

    logger.info(f"Training the model on {len(X_train)} rows...")
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed.")

    logger.info(f"Saving the trained model to {config.MODEL_PATH}...")
    joblib.dump(pipeline, config.MODEL_PATH)
    logger.info("Model saved successfully.")
