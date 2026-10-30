"""
Training pipeline for PyTorch Bidirectional LSTM (BiLSTM) emotion classification.
Trains on academic_emotions.csv, validates, and saves trained weights to models/trained/.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from training.preprocessing import Vocabulary, TextDataset, EMOTION_TO_LABEL, LABEL_TO_EMOTION
from models.bilstm_model import BiLSTMClassifier
from utils.config import DATASET_PATH, BILSTM_MODEL_PATH, VOCAB_PATH, TRAINED_DIR

def train_model(
    epochs: int = 25,
    batch_size: int = 16,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 2,
    lr: float = 0.001,
    weight_decay: float = 1e-4
):
    print("=" * 60)
    print("🚀 Starting BiLSTM Emotion Classifier Training Pipeline")
    print("=" * 60)

    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    print(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")
    
    # 2. Build Vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(train_df["text"].tolist(), min_freq=1)
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    vocab.save(str(VOCAB_PATH))
    print(f"Built and saved vocabulary with {len(vocab)} unique tokens to {VOCAB_PATH}")

    # 3. Create Datasets & DataLoaders
    train_labels = [EMOTION_TO_LABEL[e] for e in train_df["emotion"]]
    val_labels = [EMOTION_TO_LABEL[e] for e in val_df["emotion"]]
    
    train_dataset = TextDataset(train_df["text"].tolist(), train_labels, vocab, max_len=32)
    val_dataset = TextDataset(val_df["text"].tolist(), val_labels, vocab, max_len=32)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=5,
        dropout=0.3,
        pad_idx=vocab.pad_idx
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_val_acc = 0.0

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in LSTM
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_train_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += len(y_batch)

        train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                
                total_val_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == y_batch).sum().item()
                total_val += len(y_batch)

        val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, str(BILSTM_MODEL_PATH))

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

    print(f"✅ BiLSTM Training Finished! Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc*100:.1f}%")
    print(f"Model saved to {BILSTM_MODEL_PATH}")

if __name__ == "__main__":
    train_model(epochs=30)
