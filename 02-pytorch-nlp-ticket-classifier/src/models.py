import torch
import torch.nn as nn
from src.config import PAD_IDX, VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES

class EmbeddingAverageClassifier(nn.Module):
    """
    Architecture:
        input_ids [B, L]
        -> Embedding Layer [B, L, D]
        -> masked average pooling [B, D]
        -> Linear Layer [B, num_classes]
        -> logits [B, num_classes]
    
    B = batch size, L = sequence length (256), D = embedding dimension (128)
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, embedding_dim: int = EMBEDDING_DIM, 
                 num_classes: int = NUM_CLASSES, pad_idx: int = PAD_IDX):
        # Gradient for PAD is not updated

        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, L] - batch of tokenized sequences
        """
        embedded = self.embedding(input_ids)        # [B, L, D]

        # Create a mask for non-padding tokens
        mask = (input_ids != PAD_IDX)       # [B, L] dtype=bool

        # Expand mask to match the embedding dimensions
        mask_expanded = mask.unsqueeze(-1).float()      # [B, L, 1]

        # Reducing vector on PAD tokens by multiplying with the mask
        masked_embeddings = embedded * mask_expanded     # [B, L, D]

        # Sum the vector of real tokens
        sum_embeddings = masked_embeddings.sum(dim=1)    # [B, D]

        # Count the number of real tokens for each sequence (minimum 1 to avoid division by zero)
        token_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1)     # [B, 1]

        # Average the embeddings by dividing by the number of real tokens
        pooled = sum_embeddings / token_counts     # [B, D]

        logits = self.classifier(pooled)          # [B, num_classes]
        return logits