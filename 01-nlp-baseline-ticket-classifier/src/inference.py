import os
import joblib
import logging
import src.config as config
import json
from nltk.stem.snowball import SnowballStemmer
from src.preprocessing import clean_text, stem_text

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_MODEL = None

def load_model():
    """Load the trained model from disk."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Model file not found at {config.MODEL_PATH}. Please train the model first.")
        raise FileNotFoundError(f"Model file not found at {config.MODEL_PATH}. Please train the model first.")
    
    logger.info(f"Loading the trained model from {config.MODEL_PATH}...")
    _MODEL = joblib.load(config.MODEL_PATH)
    logger.info("Model loaded successfully.")
    return _MODEL

def predict_ticket(text: str | None = None, language: str | None = None):
    """Predict the category of a support ticket based on its text."""
    model = load_model()

    if not text or text.strip() == "":
        logger.error("No input text provided for prediction.")
        raise ValueError("No input text provided for prediction.")

    try:
        with open(config.LANGUAGES_PATH, 'r', encoding='utf-8') as file:
            languages = json.load(file)
    except FileNotFoundError:
        logger.error(f"Languages file not found at {config.LANGUAGES_PATH}. Language warnings will be disabled.")
        languages = {}

    if languages: 
        if not language:
            logger.warning("No language specified. Prediction could be less accurate.")
        elif language not in languages:
            logger.warning(f"Language '{language}' is not supported. Use one of: {list(languages.keys())}")

    logger.info(f"Making prediction for the input text: {text[:50]}...")
    text = clean_text(text)
    stemmer = SnowballStemmer(languages[language]) if language in languages else None
    if stemmer:
        text = stem_text(text, stemmer)
    prediction = model.predict([text])[0]
    logger.info(f"Predicted category: {prediction}")
    
    return prediction