import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('../'))
from src.config import PAD_IDX, UNK_IDX, MAX_SEQUENCE_LENGTH
from src.dataset import tokenize, build_vocab, encode_text

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
