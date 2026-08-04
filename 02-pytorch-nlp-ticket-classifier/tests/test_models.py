import torch
import pytest
from src.models import EmbeddingAverageClassifier
from src.config import PAD_IDX, VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES


@pytest.fixture
def model():
    return EmbeddingAverageClassifier()

def test_forward_pass_output_shape(model):
    """Batch [4, 256] → logits [4, 10]"""
    input_ids = torch.randint(0, VOCAB_SIZE, (4, 256))  # Random input tensor
    logits = model(input_ids)
    assert logits.shape == torch.Size([4, NUM_CLASSES])

def test_forward_pass_all_padding(model):
    """PAD batching na crash and give shape [4, 10]"""
    input_ids = torch.zeros(4, 256, dtype=torch.long)  # only PAD_IDX=0
    logits = model(input_ids)
    assert logits.shape == torch.Size([4, NUM_CLASSES])
    assert not torch.isnan(logits).any()                # no NaN

def test_forward_pass_single_token(model):
    """Ticket with only one real token — padding_idx does not mess up the average"""
    input_ids = torch.zeros(2, 256, dtype=torch.long)
    input_ids[:, 0] = 5   # only the first position is a real token
    logits = model(input_ids)
    assert logits.shape == torch.Size([2, NUM_CLASSES])

def test_forward_pass_no_grad(model):
    """Forward pass works in inference mode (no_grad)"""
    input_ids = torch.randint(0, VOCAB_SIZE, (4, 256))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == torch.Size([4, NUM_CLASSES])

def test_embedding_layer_params(model):
    """Embedding layer has the correct dimensions"""
    weight = model.embedding.weight
    assert weight.shape == torch.Size([VOCAB_SIZE, EMBEDDING_DIM])

def test_pad_embedding_is_zero(model):
    """padding_idx=0 means, that the PAD vector in the embedding is zero"""
    pad_vector = model.embedding.weight[PAD_IDX]
    assert torch.all(pad_vector == 0)