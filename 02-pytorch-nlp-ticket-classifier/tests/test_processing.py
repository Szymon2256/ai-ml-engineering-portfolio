import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('../'))
from src.dataset import tokenize, build_vocab

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
