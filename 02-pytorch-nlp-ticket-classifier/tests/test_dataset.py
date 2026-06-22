import pytest
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('../'))
from src.config import PAD_IDX, UNK_IDX, MAX_SEQUENCE_LENGTH
from src.dataset import tokenize, build_vocab, encode_text, TicketDataset, label_mapping

# ---------- Tokenizer ----------
def test_tokenize_basic():
    assert tokenize("hello world") == ["hello", "world"]

def test_tokenize_empty():
    assert tokenize("") == []

def test_tokenize_none():
    assert tokenize(None) == []

def test_tokenize_whitespace_only():
    assert tokenize("   ") == []

# ---------- Vocabulary builder ----------
def test_vocab_pad_unk_at_correct_indices():
    texts = ["hello world", "hello there"]
    vocab = build_vocab(texts, max_vocab_size=10)
    assert vocab["<PAD>"] == 0
    assert vocab["<UNK>"] == 1

def test_vocab_size_respect_limit():
    texts = ["a b c d e f g h i j k l m"]
    vocab = build_vocab(texts, max_vocab_size=5)
    assert len(vocab) == 5

def test_vocab_build_from_train_only():
    train_texts = ["apple banana"]
    vocab = build_vocab(train_texts, max_vocab_size=100)
    assert "apple" in vocab
    assert "banana" in vocab
    assert "orange" not in vocab

# ---------- Text encoding ----------
def test_encode_text_length_always_max_len():
    vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3}
    result = encode_text("hello world", vocab, max_length=10)
    assert len(result) == 10

def test_encode_text_padding():
    vocab = {"<PAD>": 0, "<UNK>": 1, "hi": 2}
    result = encode_text("hi", vocab, max_length=5)
    assert result == [2, 0, 0, 0, 0]   # hi + 4x PAD

def test_encode_text_truncation():
    vocab = {"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3, "c": 4}
    result = encode_text("a b c", vocab, max_length=2)
    assert result == [2, 3]             

def test_encode_text_unk_for_unknown_token():
    vocab = {"<PAD>": 0, "<UNK>": 1, "known": 2}
    result = encode_text("known unknown_word", vocab, max_length=5)
    assert result[0] == 2              # "known" -> 2
    assert result[1] == UNK_IDX        # "unknown_word" -> 1

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