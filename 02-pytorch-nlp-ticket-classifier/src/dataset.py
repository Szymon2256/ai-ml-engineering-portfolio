import sys
import os
sys.path.append(os.path.abspath('../'))
from collections import Counter
from pathlib import Path
import pandas as pd
import src.config as config
from src.preprocessing import split_data
import torch
import json
from torch.utils.data import DataLoader, Dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer."""
    if not isinstance(text, str):
        return []
    return text.split()

def build_vocab(texts: list[str], max_vocab_size: int = config.VOCAB_SIZE) -> dict[str, int]:
    """ Build a vocabulary mapping from tokens to indices. """
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    
    most_common = counter.most_common(max_vocab_size - 2)  # Reserve 2 for PAD and UNK

    vocab = {config.PAD_TOKEN: config.PAD_IDX, config.UNK_TOKEN: config.UNK_IDX}
    for rank, (token, _) in enumerate(most_common):
        vocab[token] = rank + 2
    return vocab

def label_mapping(labels: list[str]) -> tuple[dict, dict]:
    """
    Create mappings from label strings to indices and vice versa.
    Sorted guarantees consistent mapping across runs.
    """
    classes = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label

def encode_text(text: str, vocab: dict[str, int], max_length: int = config.MAX_SEQUENCE_LENGTH) -> list[int]:
    """
    Encode a text string into a list of token indices based on the provided vocabulary.
    """
    tokens = tokenize(text)
    ids = [vocab.get(t, config.UNK_IDX) for t in tokens]
    ids = ids[:max_length]  # Truncate if too long
    ids += [config.PAD_IDX] * (max_length - len(ids)) # Pad if too short
    return ids

class TicketDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: dict[str, int], label_to_idx: dict[str, int], max_len: int = config.MAX_SEQUENCE_LENGTH):
        self.vocab = vocab
        self.label_to_idx = label_to_idx
        self.max_len = max_len
        # Pre-encode all texts and labels for faster access during training
        self.encoded_texts = [encode_text(text, vocab, max_len) for text in df[config.TEXT_COL].tolist()]
        self.labels = [label_to_idx[label] for label in df[config.TARGET_COL].tolist()]

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs_ids = torch.tensor(self.encoded_texts[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs_ids, label

#---------------------------------------------------------
# Some additional utility functions for saving and loading vocab and label maps   
def save_vocab(vocab: dict, path: Path = config.VOCAB_PATH) -> None:
    """Save the vocabulary to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def load_vocab(path: Path = config.VOCAB_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_label_map(label_to_idx: dict, path: Path = config.LABEL_MAP_PATH) -> None:
    """Save the label mapping to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=2)

def save_reverse_label_map(idx_to_label: dict, path: Path = config.REVERSE_LABEL_MAP_PATH) -> None:
    """Save the reverse label mapping to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, ensure_ascii=False, indent=2)

def load_label_map(path: Path = config.LABEL_MAP_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_reverse_label_map(path: Path = config.REVERSE_LABEL_MAP_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
#---------------------------------------------------------

def build_data_loader(full_train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoader objects for training, validation, and testing datasets.
    """
    train_df, val_df = split_data(full_train_df, target_column=config.TARGET_COL, test_size=config.VALIDATION_SIZE, random_state=config.RANDOM_STATE)

    logger.info(f"Train:      {len(train_df):,}")
    logger.info(f"Validation: {len(val_df):,}")
    logger.info(f"Test:       {len(test_df):,}")

    vocab = build_vocab(train_df[config.TEXT_COL].tolist())
    logger.info(f"Vocabulary size: {len(vocab):,}")

    label_to_idx, idx_to_label = label_mapping(train_df[config.TARGET_COL].tolist())
    logger.info(f"Classes ({len(label_to_idx)}): {list(label_to_idx.keys())}")

    save_vocab(vocab)
    save_label_map(label_to_idx)
    save_reverse_label_map(idx_to_label)
    train_dataset = TicketDataset(train_df, vocab, label_to_idx)
    val_dataset   = TicketDataset(val_df, vocab, label_to_idx)
    test_dataset  = TicketDataset(test_df, vocab, label_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader