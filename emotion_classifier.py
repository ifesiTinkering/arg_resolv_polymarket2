#!/usr/bin/env python3
"""
Lightweight Emotion Classifier for Argument Resolver
Runs on Raspberry Pi to analyze speaker emotions
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

# Emotion labels
EMOTIONS = ["calm", "confident", "defensive", "dismissive", "passionate", "frustrated", "angry", "sarcastic"]

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

        return emotion_logits, uncertainty, confidence


class EmotionAnalyzer:
    """High-level interface for emotion analysis"""

    def __init__(self, model_path="models/enhanced_argument_classifier.pt"):
        """Initialize the emotion analyzer

        Args:
            model_path: Path to the trained model file
        """
        print(f"[EMOTION] Loading emotion classifier from {model_path}...")

        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load classifier
        self.classifier = EnhancedTextClassifier()

        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location='cpu')
            self.classifier.load_state_dict(state_dict)
            self.classifier.eval()
            print("[EMOTION] Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    def analyze(self, text):
        """Analyze text and return emotion, uncertainty, and confidence

        Args:
            text: Input text to analyze

        Returns:
            dict with keys: emotion, emotion_confidence, uncertainty, confidence
        """
        if not text or len(text.strip()) < 5:
            return {
                "emotion": "unknown",
                "emotion_confidence": 0.0,
                "uncertainty": 0.0,
                "confidence": 0.0
            }

        # Get embedding
        with torch.no_grad():
            embedding = self.embedding_model.encode(text, convert_to_tensor=True)
            embedding = embedding.unsqueeze(0)  # Add batch dimension
            embedding = embedding.cpu()  # Ensure on CPU to match model

            # Run classifier
            emotion_logits, uncertainty, confidence = self.classifier(embedding)

            # Get emotion prediction
            emotion_probs = torch.softmax(emotion_logits, dim=1)
            emotion_idx = torch.argmax(emotion_probs, dim=1).item()
            emotion_conf = emotion_probs[0, emotion_idx].item()

            emotion_label = EMOTIONS[emotion_idx]
            uncertainty_val = uncertainty.item()
            confidence_val = confidence.item()

        return {
            "emotion": emotion_label,
            "emotion_confidence": round(emotion_conf, 3),
            "uncertainty": round(uncertainty_val, 3),
            "confidence": round(confidence_val, 3)
        }

    def analyze_batch(self, texts):
        """Analyze multiple texts

        Args:
            texts: List of texts to analyze

        Returns:
            List of analysis results
        """
        return [self.analyze(text) for text in texts]

    def format_analysis(self, text, max_text_len=100):
        """Analyze text and return formatted string

        Args:
            text: Input text to analyze
            max_text_len: Max length of text to show in output

        Returns:
            Formatted analysis string
        """
        result = self.analyze(text)

        text_preview = text[:max_text_len]
        if len(text) > max_text_len:
            text_preview += "..."

        output = f"""
Text: "{text_preview}"
Emotion: {result['emotion'].upper()} (confidence: {result['emotion_confidence']:.1%})
Uncertainty: {result['uncertainty']:.3f}
Confidence: {result['confidence']:.3f}
"""
        return output.strip()


# Convenience function for quick usage
def analyze_emotion(text, model_path="models/enhanced_argument_classifier.pt"):
    """Quick function to analyze a single text

    Args:
        text: Input text
        model_path: Path to model file

    Returns:
        dict with emotion analysis
    """
    analyzer = EmotionAnalyzer(model_path)
    return analyzer.analyze(text)


if __name__ == "__main__":
    # Test the emotion classifier
    print("Testing Emotion Classifier\n" + "="*60)

    analyzer = EmotionAnalyzer()

    test_cases = [
        "Let's examine the data objectively and consider both perspectives.",
        "I'm absolutely certain that this approach is correct. The evidence clearly supports it.",
        "That's not what I meant at all. You're misrepresenting my position.",
        "That's ridiculous. Anyone with basic knowledge knows better.",
        "We must take action now! This is absolutely critical!",
        "I've explained this three times already. Why don't you understand?",
        "You're completely wrong! This is absurd!",
        "Oh sure, because that makes perfect sense. Great logic there."
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. {analyzer.format_analysis(text)}")
