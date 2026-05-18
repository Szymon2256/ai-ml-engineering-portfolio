import sys
import os
sys.path.append(os.path.abspath('../'))
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import logging
from nltk.stem.snowball import SnowballStemmer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_text(text):
    """
    Clean the input text by removing special characters, extra spaces, and converting to lowercase
    to prepare it for NLP tasks such as tokenization, vectorization and using TF-IDF. 
    This function can be extended to include more complex cleaning steps such as removing stop words, lemmatization, etc.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove HTML tags (also names, emails, acc_num, phone numbers, etc.)
    text = re.sub(re.compile('<.*?>'), '', text)

    # 3. Remove special characters ..., ,, !, ?, etc.
    text = re.sub(r'[^\w\s]', ' ', text)

    # 4. Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def stem_text(text: str, stemmer: SnowballStemmer) -> str:
    """
    Apply stemming to the input text using Snowball Stemmer.
    This function can be used to reduce words to their root form, which can help in improving the performance of NLP models.
    """
    if not isinstance(text, str):
        return ""

    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = ' '.join(stemmed_tokens)
    
    return stemmed_text

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split the DataFrame into training and testing sets.
    """

    logger.info(f"Splitting data into train and test sets with test size {test_size} and random state {random_state}...")
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[target_column]
    )

    logger.info("Data splitting completed.")
    return train_df, test_df