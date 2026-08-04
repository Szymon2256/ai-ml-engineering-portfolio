import json
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH,
    EMBEDDING_MODEL_PATH, LABEL_MAP_PATH,
    LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY,
    EMBEDDING_DIM, NUM_CLASSES, VOCAB_SIZE,
    VALIDATION_SIZE, RANDOM_STATE, BATCH_SIZE,
)
from src.dataset import build_data_loader

# --- Device ---

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# --- One epoch training ---

def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """ Train the model for one epoch and return the average loss."""

    model.train()
    total_loss = 0.0
    
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)  # [B, num_classes]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(loader)
