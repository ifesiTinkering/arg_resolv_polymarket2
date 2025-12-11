#!/usr/bin/env python3
"""
Enhanced Emotion Classifier for Arguments

Expanded emotion taxonomy to capture nuanced argumentation styles:
1. calm - Measured, rational, composed
2. confident - Assertive, self-assured
3. defensive - Protective, justifying
4. dismissive - Condescending, belittling
5. passionate - Enthusiastic, energetic (positive)
6. frustrated - Annoyed, impatient
7. angry - Heated, hostile
8. sarcastic - Mocking, ironic

This provides much richer analysis of argument dynamics!
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import asyncio
import fastapi_poe as fp
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()
POE_API_KEY = os.getenv("POE_API_KEY")

# Emotion taxonomy with linguistic markers
EMOTIONS = {
    "calm": {
        "description": "Measured, rational, composed tone",
        "markers": ["let's consider", "objectively", "rationally", "calmly speaking", "to be fair"],
        "example": "Let's objectively examine the evidence and consider both perspectives carefully."
    },
    "confident": {
        "description": "Assertive, self-assured, certain",
        "markers": ["definitely", "clearly", "obviously", "without a doubt", "I know"],
        "example": "I'm absolutely certain this is the correct approach. The data clearly supports my position."
    },
    "defensive": {
        "description": "Protective, justifying, explaining",
        "markers": ["but I", "actually", "in my defense", "you don't understand", "that's not what I meant"],
        "example": "Actually, that's not what I meant. In my defense, I was simply trying to explain the nuances."
    },
    "dismissive": {
        "description": "Condescending, belittling, ignoring",
        "markers": ["that's ridiculous", "come on", "really?", "you can't be serious", "whatever"],
        "example": "That's ridiculous. Come on, anyone with basic knowledge would know that's not how it works."
    },
    "passionate": {
        "description": "Enthusiastic, energetic, emotionally engaged (positive)",
        "markers": ["I truly believe", "this is crucial", "it's vital that", "we must", "absolutely essential"],
        "example": "I truly believe this is absolutely crucial for our future. We must take action now!"
    },
    "frustrated": {
        "description": "Annoyed, impatient, exasperated",
        "markers": ["I've already explained", "how many times", "why don't you get it", "this is pointless", "ugh"],
        "example": "I've already explained this three times. Why don't you get it? This conversation is pointless."
    },
    "angry": {
        "description": "Heated, hostile, aggressive",
        "markers": ["you're wrong", "that's absurd", "I'm done", "this is nonsense", "you're being"],
        "example": "You're completely wrong! That's absolutely absurd and frankly insulting. I'm done with this."
    },
    "sarcastic": {
        "description": "Mocking, ironic, insincere",
        "markers": ["oh sure", "yeah right", "of course", "wow, great point", "how brilliant"],
        "example": "Oh sure, because that makes perfect sense. Wow, what a brilliant argument you've constructed there."
    }
}

# Uncertainty markers
UNCERTAINTY_MARKERS = [
    "i think", "i believe", "maybe", "probably", "possibly",
    "i'm not sure", "it seems like", "i guess", "presumably",
    "in my opinion", "i suppose", "might be", "could be", "perhaps"
]

# Confidence markers
CONFIDENCE_MARKERS = [
    "i know", "definitely", "certainly", "obviously", "clearly",
    "without a doubt", "absolutely", "i'm certain", "it's a fact",
    "undeniably", "for sure", "guaranteed", "no question"
]

class EnhancedTextClassifier(nn.Module):
    """Classifier for rich argument analysis"""

    def __init__(self, embedding_dim=384, hidden_dim=128, num_emotions=8):
        super().__init__()

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Task-specific heads
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_emotions)
        )

    def forward(self, embeddings):
        shared_features = self.shared(embeddings)

        uncertainty = self.uncertainty_head(shared_features).squeeze(-1)
        confidence = self.confidence_head(shared_features).squeeze(-1)
        emotion_logits = self.emotion_head(shared_features)

        return {
            "uncertainty": uncertainty,
            "confidence": confidence,
            "emotion_logits": emotion_logits
        }

async def generate_labeled_samples_enhanced(num_samples=200):
    """Generate diverse argument samples with rich emotion labels"""

    print(f"🎯 Generating {num_samples} labeled samples across 8 emotions...")
    print(f"\n📊 Emotion Categories:")
    for i, (emotion, info) in enumerate(EMOTIONS.items(), 1):
        print(f"   {i}. {emotion.upper():12} - {info['description']}")

    topics = [
        "climate change", "AI regulation", "universal basic income",
        "space exploration", "remote work", "electric vehicles",
        "cryptocurrency", "social media age limits", "nuclear energy",
        "genetic engineering", "privacy vs security", "college education",
        "immigration policy", "minimum wage", "healthcare system",
        "gun control", "free speech online", "animal rights"
    ]

    dataset = []
    emotion_names = list(EMOTIONS.keys())

    # Ensure balanced distribution
    samples_per_emotion = num_samples // len(emotion_names)
    extra_samples = num_samples % len(emotion_names)

    for emotion_idx, emotion_name in enumerate(emotion_names):
        n_samples = samples_per_emotion + (1 if emotion_idx < extra_samples else 0)

        print(f"\n{emotion_name.upper()} ({n_samples} samples):")

        for i in range(n_samples):
            topic = np.random.choice(topics)

            # Determine style (uncertainty/confidence) based on emotion
            if emotion_name in ["calm", "defensive"]:
                style = np.random.choice(["uncertain", "balanced"])
            elif emotion_name in ["confident", "dismissive", "angry"]:
                style = "confident"
            elif emotion_name in ["passionate", "frustrated"]:
                style = np.random.choice(["confident", "balanced"])
            else:  # sarcastic
                style = np.random.choice(["uncertain", "confident", "balanced"])

            # Generate sample
            prompt = f"""Generate a brief argument statement (2-3 sentences) about {topic}.

Emotion: {emotion_name} - {EMOTIONS[emotion_name]['description']}
Example markers to use: {', '.join(EMOTIONS[emotion_name]['markers'][:3])}
Style: {style}

Make it sound natural and realistic, like something someone would actually say in an argument.
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
                text_lower = text.lower()

                # Count markers
                uncertainty_count = sum(1 for marker in UNCERTAINTY_MARKERS if marker in text_lower)
                confidence_count = sum(1 for marker in CONFIDENCE_MARKERS if marker in text_lower)

                # Normalize to 0-1
                uncertainty_score = min(uncertainty_count / 3.0, 1.0)
                confidence_score = min(confidence_count / 3.0, 1.0)

                dataset.append({
                    "id": f"{emotion_name}_{i:03d}",
                    "text": text,
                    "topic": topic,
                    "emotion_name": emotion_name,
                    "emotion_label": emotion_idx,
                    "uncertainty_score": uncertainty_score,
                    "confidence_score": confidence_score,
                    "marker_counts": {
                        "uncertainty": uncertainty_count,
                        "confidence": confidence_count
                    }
                })

                print(f"  [{i+1}/{n_samples}] Generated (U:{uncertainty_count} C:{confidence_count})")

            except Exception as e:
                print(f"  [{i+1}/{n_samples}] Failed: {e}")

            # Rate limiting
            await asyncio.sleep(0.5)

    # Save dataset
    output_file = Path("training_data/enhanced_labeled_arguments.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Dataset saved: {output_file}")
    print(f"   Total samples: {len(dataset)}")

    # Statistics
    print(f"\n📊 Dataset Statistics:")
    for emotion_name in emotion_names:
        count = sum(1 for s in dataset if s['emotion_name'] == emotion_name)
        print(f"   {emotion_name:12}: {count} samples")

    print(f"\n   Avg uncertainty: {np.mean([s['uncertainty_score'] for s in dataset]):.2f}")
    print(f"   Avg confidence: {np.mean([s['confidence_score'] for s in dataset]):.2f}")

    # Save test examples
    test_examples = {}
    for emotion_name in emotion_names:
        examples = [s for s in dataset if s['emotion_name'] == emotion_name]
        if examples:
            test_examples[emotion_name] = {
                "example": examples[0]['text'],
                "description": EMOTIONS[emotion_name]['description']
            }

    test_file = Path("training_data/test_examples.json")
    with open(test_file, 'w') as f:
        json.dump(test_examples, f, indent=2)

    print(f"\n📝 Test examples saved: {test_file}")

    return dataset

class EnhancedArgumentDataset(Dataset):
    """Dataset for enhanced classifier"""

    def __init__(self, data_path, encoder_model):
        with open(data_path) as f:
            self.data = json.load(f)
        self.encoder = encoder_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        embedding = self.encoder.encode(entry["text"], convert_to_tensor=True)

        return {
            "embedding": embedding,
            "uncertainty": torch.tensor(entry["uncertainty_score"], dtype=torch.float32),
            "confidence": torch.tensor(entry["confidence_score"], dtype=torch.float32),
            "emotion": torch.tensor(entry["emotion_label"], dtype=torch.long)
        }

def train_enhanced_classifier(data_path, output_dir="models", epochs=50, batch_size=16):
    """Train the enhanced classifier"""

    print("🔧 Loading sentence transformer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    encoder = encoder.to(device)

    print("📊 Loading dataset...")
    dataset = EnhancedArgumentDataset(data_path, encoder)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Initialize classifier
    print("🧠 Initializing classifier...")
    classifier = EnhancedTextClassifier(embedding_dim=384, num_emotions=8).to(device)

    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\n🚀 Training for {epochs} epochs...\n")

    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "emotion_acc": []}

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

            # Weighted combination (emotion is most important)
            loss = 0.3 * uncertainty_loss + 0.3 * confidence_loss + 0.4 * emotion_loss

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

                loss = 0.3 * uncertainty_loss + 0.3 * confidence_loss + 0.4 * emotion_loss

                val_loss += loss.item() * embeddings.size(0)
                val_samples += embeddings.size(0)

                emotion_preds = outputs["emotion_logits"].argmax(dim=1)
                all_emotion_preds.extend(emotion_preds.cpu().numpy())
                all_emotion_targets.extend(emotion_target.cpu().numpy())

        val_loss /= val_samples
        emotion_acc = accuracy_score(all_emotion_targets, all_emotion_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["emotion_acc"].append(emotion_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}/{epochs} - "
              f"Train: {train_loss:.4f}, "
              f"Val: {val_loss:.4f}, "
              f"Emotion Acc: {emotion_acc:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(output_dir, exist_ok=True)
            torch.save(classifier.state_dict(), f"{output_dir}/enhanced_argument_classifier.pt")
            print(f"  ✅ Saved (val_loss: {val_loss:.4f})")

    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")

    # Final evaluation
    emotion_names = list(EMOTIONS.keys())
    print(f"\n📊 Final Evaluation:")
    print(classification_report(all_emotion_targets, all_emotion_preds,
                                target_names=emotion_names, digits=3))

    # Confusion matrix
    cm = confusion_matrix(all_emotion_targets, all_emotion_preds)
    print(f"\n🔍 Confusion Matrix:")
    print(f"   Rows: True, Cols: Predicted")
    print(f"   {' '.join([e[:4] for e in emotion_names])}")
    for i, row in enumerate(cm):
        print(f"   {emotion_names[i][:4]} {row}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        num_samples = 200
        if len(sys.argv) > 2:
            num_samples = int(sys.argv[2])

        asyncio.run(generate_labeled_samples_enhanced(num_samples))

    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        data_path = "training_data/enhanced_labeled_arguments.json"

        if not Path(data_path).exists():
            print(f"❌ Dataset not found: {data_path}")
            print("   Run: python enhanced_emotion_classifier.py generate 200")
            sys.exit(1)

        train_enhanced_classifier(data_path, epochs=50)

    else:
        print("Usage:")
        print("  python enhanced_emotion_classifier.py generate [num_samples]")
        print("  python enhanced_emotion_classifier.py train")
        print("\nEmotions detected:")
        for i, (emotion, info) in enumerate(EMOTIONS.items(), 1):
            print(f"  {i}. {emotion:12} - {info['description']}")
