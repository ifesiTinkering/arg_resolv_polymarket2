#!/usr/bin/env python3
"""
Linear Probing Approach for Argument Analysis

Instead of fine-tuning Whisper (expensive, slow), we:
1. Use Whisper as-is for transcription
2. Extract embeddings from Whisper's encoder
3. Train a lightweight classifier on top to detect:
   - Uncertainty level (0-1 score)
   - Confidence level (0-1 score)
   - Emotional intensity (calm/medium/heated)

This is much faster and shows "thoughtful system-AI co-design"!
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import whisper
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Set torchaudio backend
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

class ArgumentClassifier(nn.Module):
    """
    Lightweight classifier on top of Whisper embeddings

    Input: Whisper encoder embeddings (1500 x 384 for tiny model)
    Output:
        - uncertainty_score (0-1)
        - confidence_score (0-1)
        - emotional_intensity (0=calm, 1=medium, 2=heated)
    """

    def __init__(self, embedding_dim=384, hidden_dim=128):
        super().__init__()

        # Shared layers
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool over time dimension

        # Separate heads for each task
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output 0-1
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output 0-1
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3),  # 3 classes: calm, medium, heated
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: (batch, time, embedding_dim) from Whisper encoder

        Returns:
            dict with uncertainty, confidence, emotion predictions
        """

        # Pool over time: (batch, embedding_dim, time) -> (batch, embedding_dim, 1)
        pooled = self.pool(embeddings.transpose(1, 2)).squeeze(-1)  # (batch, embedding_dim)

        uncertainty = self.uncertainty_head(pooled).squeeze(-1)  # (batch,)
        confidence = self.confidence_head(pooled).squeeze(-1)    # (batch,)
        emotion_logits = self.emotion_head(pooled)                # (batch, 3)

        return {
            "uncertainty": uncertainty,
            "confidence": confidence,
            "emotion_logits": emotion_logits
        }

class AudioArgumentDataset(Dataset):
    """Dataset for training the linear probe classifier"""

    def __init__(self, manifest_path, whisper_model):
        """
        Args:
            manifest_path: Path to training_manifest.json
            whisper_model: Loaded Whisper model for extracting embeddings
        """
        with open(manifest_path) as f:
            self.data = json.load(f)

        self.whisper_model = whisper_model

        # Emotion mapping
        self.emotion_map = {"calm": 0, "medium": 1, "heated": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Load audio and get Whisper embeddings
        audio_path = entry["audio_path"]
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # Get mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)

        # Extract encoder embeddings (don't decode, just get embeddings)
        with torch.no_grad():
            embeddings = self.whisper_model.encoder(mel.unsqueeze(0))  # (1, time, embedding_dim)

        # Get labels from metadata
        metadata = entry["metadata"]

        # Normalize counts to 0-1 scores
        uncertainty_score = min(metadata["uncertainty_count"] / 5.0, 1.0)  # Cap at 5 markers
        confidence_score = min(metadata["confidence_count"] / 5.0, 1.0)

        emotion_label = self.emotion_map[metadata["emotional_intensity"]]

        return {
            "embeddings": embeddings.squeeze(0),  # (time, embedding_dim)
            "uncertainty": torch.tensor(uncertainty_score, dtype=torch.float32),
            "confidence": torch.tensor(confidence_score, dtype=torch.float32),
            "emotion": torch.tensor(emotion_label, dtype=torch.long)
        }

def train_classifier(manifest_path, output_dir="models", epochs=20, batch_size=8):
    """
    Train the linear probe classifier

    Args:
        manifest_path: Path to training_manifest.json
        output_dir: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
    """

    print("🔧 Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("tiny.en", device=device)

    print("📊 Loading dataset...")
    dataset = AudioArgumentDataset(manifest_path, whisper_model)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Initialize classifier
    print("🧠 Initializing classifier...")
    embedding_dim = 384  # Whisper tiny.en embedding dimension
    classifier = ArgumentClassifier(embedding_dim=embedding_dim).to(device)

    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    print(f"\n🚀 Training for {epochs} epochs...\n")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        classifier.train()
        train_loss = 0
        train_samples = 0

        for batch in train_loader:
            embeddings = batch["embeddings"].to(device)
            uncertainty_target = batch["uncertainty"].to(device)
            confidence_target = batch["confidence"].to(device)
            emotion_target = batch["emotion"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = classifier(embeddings)

            # Compute losses
            uncertainty_loss = mse_loss(outputs["uncertainty"], uncertainty_target)
            confidence_loss = mse_loss(outputs["confidence"], confidence_target)
            emotion_loss = ce_loss(outputs["emotion_logits"], emotion_target)

            # Combined loss
            loss = uncertainty_loss + confidence_loss + emotion_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * embeddings.size(0)
            train_samples += embeddings.size(0)

        train_loss /= train_samples

        # Validation
        classifier.eval()
        val_loss = 0
        val_samples = 0
        all_emotion_preds = []
        all_emotion_targets = []

        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch["embeddings"].to(device)
                uncertainty_target = batch["uncertainty"].to(device)
                confidence_target = batch["confidence"].to(device)
                emotion_target = batch["emotion"].to(device)

                outputs = classifier(embeddings)

                uncertainty_loss = mse_loss(outputs["uncertainty"], uncertainty_target)
                confidence_loss = mse_loss(outputs["confidence"], confidence_target)
                emotion_loss = ce_loss(outputs["emotion_logits"], emotion_target)

                loss = uncertainty_loss + confidence_loss + emotion_loss

                val_loss += loss.item() * embeddings.size(0)
                val_samples += embeddings.size(0)

                # Track emotion predictions for accuracy
                emotion_preds = outputs["emotion_logits"].argmax(dim=1)
                all_emotion_preds.extend(emotion_preds.cpu().numpy())
                all_emotion_targets.extend(emotion_target.cpu().numpy())

        val_loss /= val_samples
        emotion_acc = accuracy_score(all_emotion_targets, all_emotion_preds)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Emotion Acc: {emotion_acc:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save(classifier.state_dict(), f"{output_dir}/argument_classifier.pt")
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")

    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Model saved: {output_dir}/argument_classifier.pt")

    # Final evaluation
    print(f"\n📊 Final Evaluation:")
    print(classification_report(all_emotion_targets, all_emotion_preds,
                                target_names=["calm", "medium", "heated"]))

if __name__ == "__main__":
    manifest_path = "training_data/training_manifest.json"

    if not Path(manifest_path).exists():
        print(f"❌ Training manifest not found: {manifest_path}")
        print("   Run generate_training_data.py and generate_audio.py first!")
        sys.exit(1)

    train_classifier(manifest_path, epochs=20)
