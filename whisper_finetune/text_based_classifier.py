#!/usr/bin/env python3
"""
Text-based Argument Classifier (No audio needed!)

Instead of using Whisper embeddings, we can:
1. Use sentence transformers to embed transcripts
2. Train classifier to detect uncertainty/confidence/emotion
3. Apply to transcribed text from Whisper

This is faster and doesn't require audio generation!
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import asyncio
import fastapi_poe as fp
from dotenv import load_dotenv

load_dotenv()
POE_API_KEY = os.getenv("POE_API_KEY")

# Uncertainty and confidence markers
UNCERTAINTY_MARKERS = [
    "i think", "i believe", "maybe", "probably", "possibly",
    "i'm not sure", "it seems like", "i guess", "presumably",
    "in my opinion", "i suppose", "might be", "could be", "perhaps"
]

CONFIDENCE_MARKERS = [
    "i know", "definitely", "certainly", "obviously", "clearly",
    "without a doubt", "absolutely", "i'm certain", "it's a fact",
    "undeniably", "for sure", "guaranteed", "no question"
]

class TextClassifier(nn.Module):
    """Classifier for argument analysis from text embeddings"""

    def __init__(self, embedding_dim=384, hidden_dim=128):
        super().__init__()

        # Heads for each task
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3),  # calm, medium, heated
        )

    def forward(self, embeddings):
        uncertainty = self.uncertainty_head(embeddings).squeeze(-1)
        confidence = self.confidence_head(embeddings).squeeze(-1)
        emotion_logits = self.emotion_head(embeddings)

        return {
            "uncertainty": uncertainty,
            "confidence": confidence,
            "emotion_logits": emotion_logits
        }

async def generate_labeled_samples(num_samples=100):
    """
    Generate synthetic argument texts with labels using LLM

    This creates training data without needing audio!
    """

    print(f"🎯 Generating {num_samples} labeled argument samples...")

    topics = [
        "climate change", "AI regulation", "universal basic income",
        "space exploration", "remote work", "electric vehicles",
        "cryptocurrency", "social media age limits", "nuclear energy",
        "genetic engineering", "privacy vs security", "college education"
    ]

    styles = ["uncertain", "confident", "balanced"]
    emotions = ["calm", "medium", "heated"]

    dataset = []

    for i in range(num_samples):
        topic = np.random.choice(topics)
        style = np.random.choice(styles, p=[0.25, 0.25, 0.5])  # More balanced
        emotion = np.random.choice(emotions, p=[0.5, 0.3, 0.2])  # More calm

        # Generate sample
        prompt = f"""Generate a brief argument statement (2-3 sentences) about {topic}.

Style: {style} ({"use uncertainty markers like 'I think', 'maybe', 'probably'" if style == "uncertain" else "use confidence markers like 'definitely', 'clearly', 'certainly'" if style == "confident" else "mix both naturally"})

Emotion: {emotion} ({"calm, measured" if emotion == "calm" else "moderately emotional" if emotion == "medium" else "heated, passionate"})

Just give the statement, nothing else."""

        try:
            message = fp.ProtocolMessage(role="user", content=prompt)
            text = ""
            async for partial in fp.get_bot_response(
                messages=[message],
                bot_name="GPT-4o-Mini",
                api_key=POE_API_KEY
            ):
                text += partial.text

            text = text.strip()

            # Compute labels
            text_lower = text.lower()

            # Count markers
            uncertainty_count = sum(1 for marker in UNCERTAINTY_MARKERS if marker in text_lower)
            confidence_count = sum(1 for marker in CONFIDENCE_MARKERS if marker in text_lower)

            # Normalize to 0-1
            uncertainty_score = min(uncertainty_count / 3.0, 1.0)
            confidence_score = min(confidence_count / 3.0, 1.0)

            # Emotion label
            emotion_map = {"calm": 0, "medium": 1, "heated": 2}
            emotion_label = emotion_map[emotion]

            dataset.append({
                "id": f"sample_{i:04d}",
                "text": text,
                "topic": topic,
                "style": style,
                "emotion_str": emotion,
                "uncertainty_score": uncertainty_score,
                "confidence_score": confidence_score,
                "emotion_label": emotion_label,
                "marker_counts": {
                    "uncertainty": uncertainty_count,
                    "confidence": confidence_count
                }
            })

            print(f"[{i+1}/{num_samples}] Generated: {topic} ({style}, {emotion}) - U:{uncertainty_count} C:{confidence_count}")

        except Exception as e:
            print(f"[{i+1}/{num_samples}] Failed: {e}")

        # Rate limiting
        await asyncio.sleep(0.5)

    # Save dataset
    output_file = Path("training_data/text_labeled_arguments.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Dataset saved: {output_file}")
    print(f"   Total samples: {len(dataset)}")

    # Statistics
    print(f"\n📊 Statistics:")
    print(f"   Avg uncertainty: {np.mean([s['uncertainty_score'] for s in dataset]):.2f}")
    print(f"   Avg confidence: {np.mean([s['confidence_score'] for s in dataset]):.2f}")
    print(f"   Calm: {sum(1 for s in dataset if s['emotion_label'] == 0)}")
    print(f"   Medium: {sum(1 for s in dataset if s['emotion_label'] == 1)}")
    print(f"   Heated: {sum(1 for s in dataset if s['emotion_label'] == 2)}")

    return dataset

class TextArgumentDataset(Dataset):
    """Dataset for training text-based classifier"""

    def __init__(self, data_path, encoder_model):
        with open(data_path) as f:
            self.data = json.load(f)

        self.encoder = encoder_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # Encode text
        embedding = self.encoder.encode(entry["text"], convert_to_tensor=True)

        return {
            "embedding": embedding,
            "uncertainty": torch.tensor(entry["uncertainty_score"], dtype=torch.float32),
            "confidence": torch.tensor(entry["confidence_score"], dtype=torch.float32),
            "emotion": torch.tensor(entry["emotion_label"], dtype=torch.long)
        }

def train_text_classifier(data_path, output_dir="models", epochs=30, batch_size=16):
    """Train the text-based classifier"""

    print("🔧 Loading sentence transformer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight
    encoder = encoder.to(device)

    print("📊 Loading dataset...")
    dataset = TextArgumentDataset(data_path, encoder)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Initialize classifier
    print("🧠 Initializing classifier...")
    embedding_dim = 384  # all-MiniLM-L6-v2 dimension
    classifier = TextClassifier(embedding_dim=embedding_dim).to(device)

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
            embeddings = batch["embedding"].to(device)
            uncertainty_target = batch["uncertainty"].to(device)
            confidence_target = batch["confidence"].to(device)
            emotion_target = batch["emotion"].to(device)

            optimizer.zero_grad()

            outputs = classifier(embeddings)

            uncertainty_loss = mse_loss(outputs["uncertainty"], uncertainty_target)
            confidence_loss = mse_loss(outputs["confidence"], confidence_target)
            emotion_loss = ce_loss(outputs["emotion_logits"], emotion_target)

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
                embeddings = batch["embedding"].to(device)
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

                emotion_preds = outputs["emotion_logits"].argmax(dim=1)
                all_emotion_preds.extend(emotion_preds.cpu().numpy())
                all_emotion_targets.extend(emotion_target.cpu().numpy())

        val_loss /= val_samples
        emotion_acc = accuracy_score(all_emotion_targets, all_emotion_preds)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train: {train_loss:.4f}, "
              f"Val: {val_loss:.4f}, "
              f"Emotion Acc: {emotion_acc:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save(classifier.state_dict(), f"{output_dir}/text_argument_classifier.pt")
            print(f"  ✅ Saved (val_loss: {val_loss:.4f})")

    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Model: {output_dir}/text_argument_classifier.pt")

    # Final evaluation
    if all_emotion_targets:
        print(f"\n📊 Final Evaluation:")
        print(classification_report(all_emotion_targets, all_emotion_preds,
                                    target_names=["calm", "medium", "heated"]))

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # Generate dataset
        num_samples = 100
        if len(sys.argv) > 2:
            num_samples = int(sys.argv[2])

        asyncio.run(generate_labeled_samples(num_samples))

    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        # Train classifier
        data_path = "training_data/text_labeled_arguments.json"

        if not Path(data_path).exists():
            print(f"❌ Dataset not found: {data_path}")
            print("   Run: python text_based_classifier.py generate 100")
            sys.exit(1)

        train_text_classifier(data_path, epochs=30)

    else:
        print("Usage:")
        print("  python text_based_classifier.py generate [num_samples]  - Generate training data")
        print("  python text_based_classifier.py train                   - Train classifier")
