#!/usr/bin/env python3
"""
Test the Enhanced Emotion Classifier

This script lets you test the classifier on:
1. Pre-generated test examples (one per emotion)
2. Your own custom text
3. Existing argument transcripts from arguments_db
"""

import os
import sys
import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from enhanced_emotion_classifier import EnhancedTextClassifier, EMOTIONS

def load_classifier(model_path="models/enhanced_argument_classifier.pt"):
    """Load the trained classifier"""

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Train the model first: python enhanced_emotion_classifier.py train")
        return None, None

    print("🔧 Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load encoder
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    encoder = encoder.to(device)

    # Load classifier
    classifier = EnhancedTextClassifier(embedding_dim=384, num_emotions=8).to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()

    print("✅ Models loaded")
    return encoder, classifier

def analyze_text(text: str, encoder, classifier):
    """Analyze a text and return predictions"""

    device = next(classifier.parameters()).device

    # Encode text
    with torch.no_grad():
        embedding = encoder.encode(text, convert_to_tensor=True).to(device)

        # Run classifier
        outputs = classifier(embedding.unsqueeze(0))

        uncertainty = outputs["uncertainty"].item()
        confidence = outputs["confidence"].item()
        emotion_logits = outputs["emotion_logits"]
        emotion_probs = torch.softmax(emotion_logits, dim=1)[0]
        emotion_idx = emotion_probs.argmax().item()

        emotion_names = list(EMOTIONS.keys())
        predicted_emotion = emotion_names[emotion_idx]
        emotion_confidence = emotion_probs[emotion_idx].item()

        # Get top 3 emotions
        top3_indices = emotion_probs.argsort(descending=True)[:3]
        top3_emotions = [(emotion_names[i], emotion_probs[i].item()) for i in top3_indices]

        return {
            "text": text,
            "uncertainty_score": uncertainty,
            "confidence_score": confidence,
            "predicted_emotion": predicted_emotion,
            "emotion_confidence": emotion_confidence,
            "top3_emotions": top3_emotions,
            "all_emotion_probs": {emotion_names[i]: emotion_probs[i].item() for i in range(len(emotion_names))}
        }

def print_analysis(result):
    """Pretty print analysis results"""

    print("\n" + "="*60)
    print("TEXT:")
    print("="*60)
    print(result["text"])
    print()

    print("="*60)
    print("ANALYSIS:")
    print("="*60)

    print(f"\n📊 Scores:")
    print(f"   Uncertainty: {result['uncertainty_score']:.3f} {'🤔' if result['uncertainty_score'] > 0.5 else ''}")
    print(f"   Confidence:  {result['confidence_score']:.3f} {'💪' if result['confidence_score'] > 0.5 else ''}")

    print(f"\n😀 Predicted Emotion: {result['predicted_emotion'].upper()} ({result['emotion_confidence']:.1%})")
    print(f"   Description: {EMOTIONS[result['predicted_emotion']]['description']}")

    print(f"\n🥇 Top 3 Emotions:")
    for i, (emotion, prob) in enumerate(result['top3_emotions'], 1):
        bar = "█" * int(prob * 20)
        print(f"   {i}. {emotion:12} {prob:.1%} {bar}")

    print("\n" + "="*60 + "\n")

def test_on_examples():
    """Test on pre-generated examples"""

    encoder, classifier = load_classifier()
    if encoder is None:
        return

    # Load test examples
    test_file = Path("training_data/test_examples.json")
    if not test_file.exists():
        print(f"❌ Test examples not found: {test_file}")
        print("   Generate data first: python enhanced_emotion_classifier.py generate 200")
        return

    with open(test_file) as f:
        examples = json.load(f)

    print("\n🧪 Testing on Pre-Generated Examples\n")

    correct = 0
    total = 0

    for true_emotion, info in examples.items():
        text = info["example"]
        result = analyze_text(text, encoder, classifier)

        print(f"\n{'─'*60}")
        print(f"True Emotion: {true_emotion.upper()}")
        print(f"{'─'*60}")
        print(f"Text: {text[:100]}...")
        print(f"Predicted: {result['predicted_emotion'].upper()} ({result['emotion_confidence']:.1%})")

        if result['predicted_emotion'] == true_emotion:
            print("✅ CORRECT")
            correct += 1
        else:
            print(f"❌ INCORRECT (expected {true_emotion})")

        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"{'='*60}")

def test_interactive():
    """Interactive testing mode"""

    encoder, classifier = load_classifier()
    if encoder is None:
        return

    print("\n🎯 Interactive Testing Mode")
    print("Type argument text to analyze (or 'quit' to exit)\n")

    while True:
        print("\n" + "─"*60)
        text = input("Enter text: ").strip()

        if not text or text.lower() == 'quit':
            break

        result = analyze_text(text, encoder, classifier)
        print_analysis(result)

def test_on_real_arguments():
    """Test on real arguments from arguments_db"""

    encoder, classifier = load_classifier()
    if encoder is None:
        return

    # Find arguments
    db_path = Path("../arguments_db/arguments")
    if not db_path.exists():
        print(f"❌ Arguments database not found: {db_path}")
        return

    argument_dirs = sorted(db_path.iterdir(), reverse=True)[:5]  # Latest 5

    print("\n🗂️  Testing on Real Arguments from Database\n")

    for arg_dir in argument_dirs:
        if not arg_dir.is_dir():
            continue

        transcript_file = arg_dir / "transcript.txt"
        if not transcript_file.exists():
            continue

        print(f"\n{'='*60}")
        print(f"Argument: {arg_dir.name}")
        print(f"{'='*60}")

        # Read transcript
        with open(transcript_file) as f:
            transcript = f.read()

        # Parse by speaker
        lines = transcript.split('\n')
        for line in lines[:3]:  # First 3 lines
            if ':' not in line:
                continue

            parts = line.split(':', 1)
            if len(parts) != 2:
                continue

            speaker = parts[0].strip()
            text = parts[1].strip()

            # Remove timestamp
            if ']' in text:
                text = text.split(']', 1)[1].strip()

            if text:
                result = analyze_text(text, encoder, classifier)
                print(f"\n{speaker}:")
                print(f"   Text: {text[:80]}...")
                print(f"   Emotion: {result['predicted_emotion']} ({result['emotion_confidence']:.1%})")
                print(f"   Uncertainty: {result['uncertainty_score']:.2f}, Confidence: {result['confidence_score']:.2f}")

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "examples":
            test_on_examples()
        elif mode == "interactive":
            test_interactive()
        elif mode == "real":
            test_on_real_arguments()
        elif mode == "text":
            # Test on provided text
            if len(sys.argv) < 3:
                print("Usage: python test_classifier.py text \"Your text here\"")
                return

            encoder, classifier = load_classifier()
            if encoder is None:
                return

            text = " ".join(sys.argv[2:])
            result = analyze_text(text, encoder, classifier)
            print_analysis(result)
        else:
            print(f"Unknown mode: {mode}")
            print("Usage:")
            print("  python test_classifier.py examples      - Test on pre-generated examples")
            print("  python test_classifier.py interactive   - Interactive testing")
            print("  python test_classifier.py real          - Test on real arguments from DB")
            print("  python test_classifier.py text \"...\"    - Test specific text")
    else:
        print("Enhanced Emotion Classifier Testing")
        print("="*60)
        print("\nUsage:")
        print("  python test_classifier.py examples      - Test on pre-generated examples")
        print("  python test_classifier.py interactive   - Interactive testing")
        print("  python test_classifier.py real          - Test on real arguments from DB")
        print("  python test_classifier.py text \"...\"    - Test specific text")
        print("\nEmotions detected:")
        for i, (emotion, info) in enumerate(EMOTIONS.items(), 1):
            print(f"  {i}. {emotion:12} - {info['description']}")

if __name__ == "__main__":
    main()
