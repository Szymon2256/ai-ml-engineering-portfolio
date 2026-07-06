import pytest
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('../'))
from src.config import PAD_IDX, UNK_IDX, MAX_SEQUENCE_LENGTH
from src.dataset import tokenize, build_vocab, encode_text, TicketDataset, label_mapping

# ---------- Dataset ----------
@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        "cleaned_context": ["cannot login account", "billing issue refund", ""],
        "queue": ["Technical Support", "Billing and Payments", "Technical Support"]
    })
    vocab = build_vocab(df["cleaned_context"].tolist(), max_vocab_size=100)
    label_to_idx, _ = label_mapping(df["queue"].tolist())
    return df, vocab, label_to_idx

def test_dataset_length(sample_data):
    df, vocab, label_to_idx = sample_data
    dataset = TicketDataset(df, vocab, label_to_idx)
    assert len(dataset) == 3

def test_dataset_item_shapes(sample_data):
    df, vocab, label_to_idx = sample_data
    dataset = TicketDataset(df, vocab, label_to_idx)
    input_ids, label = dataset[0]
    assert input_ids.shape == torch.Size([MAX_SEQUENCE_LENGTH])
    assert label.shape == torch.Size([]) 

def test_dataset_item_dtypes(sample_data):
    df, vocab, label_to_idx = sample_data
    dataset = TicketDataset(df, vocab, label_to_idx)
    input_ids, label = dataset[0]
    assert input_ids.dtype == torch.long
    assert label.dtype == torch.long

def test_dataset_empty_text_gives_all_padding(sample_data):
    df, vocab, label_to_idx = sample_data
    dataset = TicketDataset(df, vocab, label_to_idx)
    input_ids, _ = dataset[2]   
    assert input_ids.tolist() == [PAD_IDX] * MAX_SEQUENCE_LENGTH