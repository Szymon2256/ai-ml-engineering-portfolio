import json
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH,
    EMBEDDING_MODEL_PATH, LABEL_MAP_PATH,
    LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY,
    EMBEDDING_DIM, NUM_CLASSES, VOCAB_SIZE,
    VALIDATION_SIZE, RANDOM_STATE, BATCH_SIZE,
)
from src.dataset import build_data_loader, load_label_map
from src.models import EmbeddingAverageClassifier

# --- One epoch training ---

def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """ Train the model for one epoch and return the average loss."""

    model.train()
    total_loss = 0.0
    
    for input_ids, labels in loader:
        # Move data to the specified device (CPU or GPU)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass
        logits = model(input_ids)  # [B, num_classes]
        # Compute the loss
        loss = criterion(logits, labels)
        # Backward pass
        loss.backward()
        # Update the model parameters
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(loader)

# --- Evaluation on validation data or test data ---

def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> tuple:
    """ Going through the loader without weights update and return the average loss, accuracy and F1 score."""

    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad(): # No gradient computation for evaluation
        for input_ids, labels in loader:
                # Move data to the specified device (CPU or GPU)
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                # Forward pass
                logits = model(input_ids)  # [B, num_classes]
                # Compute the loss
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                # Store predictions and labels for metric calculation
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())        

    # Calculate metrics
    average_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

# --- Save the checkpoint ---

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_metrics: dict) -> None:
     """Save the model and metadata to a .pt file."""

     os.makedirs(os.path.dirname(EMBEDDING_MODEL_PATH), exist_ok=True)  # Ensure the directory exists

     checkpoint = {
        "epoch":                  epoch,
        "model_state_dict":       model.state_dict(),
        "optimizer_state_dict":   optimizer.state_dict(),
        "val_loss":               val_metrics["loss"],
        "val_accuracy":           val_metrics["accuracy"],
        "val_macro_f1":           val_metrics["macro_f1"],
        "val_weighted_f1":        val_metrics["weighted_f1"],
        "hyperparameters": {
            "vocab_size":     VOCAB_SIZE,
            "embedding_dim":  EMBEDDING_DIM,
            "num_classes":    NUM_CLASSES,
            "learning_rate":  LEARNING_RATE,
            "batch_size":     BATCH_SIZE,
            "num_epochs":     NUM_EPOCHS,
        }
    }

     torch.save(checkpoint, EMBEDDING_MODEL_PATH)
     print(f" --> Checkpoint saved at epoch {epoch}")

# --- Main function ---

def main():

    # Set the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    train_loader, val_loader, test_loader = build_data_loader(train_df, test_df)

    print(f"Train batches:      {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches:       {len(test_loader)}")

    # Create a model instance
    model = EmbeddingAverageClassifier().to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute class weights to handle class imbalance
    label_to_idx = load_label_map()

    train_labels_numeric = [label_to_idx[label] for label in train_df["queue"].tolist()]

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=train_labels_numeric
    )

    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"\nClass weights: {[round(w, 3) for w in weights]}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...\n")
    best_macro_f1 = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_acc: {val_metrics['accuracy']:.4f} | "
            f"val_macro_f1: {val_metrics['macro_f1']:.4f} | "
            f"val_weighted_f1: {val_metrics['weighted_f1']:.4f}"
        )

        # Save checkpoint if the macro F1 score improves
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(model, optimizer, epoch, val_metrics)

    # Load the best model for final evaluation on the test set
    print("\nLoading best checkpoint...")
    checkpoint = torch.load(EMBEDDING_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\n=== Test Results ===")
    print(f"Test accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Test weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"Best epoch:       {checkpoint['epoch']}")

if __name__ == "__main__":
    main()