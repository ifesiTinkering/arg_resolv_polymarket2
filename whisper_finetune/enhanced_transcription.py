#!/usr/bin/env python3
"""
Enhanced transcription using Whisper + Linear Probe Classifier

This module augments standard Whisper transcription with:
- Uncertainty detection ("I think", "maybe")
- Confidence detection ("definitely", "certainly")
- Emotional intensity (calm/medium/heated)

To use in your argument resolver:
    from enhanced_transcription import enhanced_transcribe

    result = enhanced_transcribe("audio.wav")
    # Returns: {
    #     "text": "standard transcription",
    #     "uncertainty_score": 0.75,  # 0-1
    #     "confidence_score": 0.45,   # 0-1
    #     "emotional_intensity": "heated",
    #     "analysis": "Speaker shows high uncertainty with phrases like..."
    # }
"""

import os
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

import torch
import torch.nn as nn
import whisper
from pathlib import Path

# Import the classifier architecture
from linear_probe_approach import ArgumentClassifier

# Global cache
_WHISPER_MODEL = None
_CLASSIFIER = None

def load_models(whisper_model_name="tiny.en", classifier_path="models/argument_classifier.pt"):
    """Load Whisper and classifier models"""
    global _WHISPER_MODEL, _CLASSIFIER

    if _WHISPER_MODEL is None:
        print("[INFO] Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _WHISPER_MODEL = whisper.load_model(whisper_model_name, device=device)

    if _CLASSIFIER is None and Path(classifier_path).exists():
        print("[INFO] Loading argument classifier...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CLASSIFIER = ArgumentClassifier(embedding_dim=384).to(device)
        _CLASSIFIER.load_state_dict(torch.load(classifier_path, map_location=device))
        _CLASSIFIER.eval()

    return _WHISPER_MODEL, _CLASSIFIER

def enhanced_transcribe(audio_path: str) -> dict:
    """
    Transcribe audio with enhanced argument analysis

    Args:
        audio_path: Path to audio file

    Returns:
        dict with transcription and analysis:
        {
            "text": "transcription",
            "uncertainty_score": 0.75,
            "confidence_score": 0.45,
            "emotional_intensity": "heated",
            "analysis": "Detailed analysis...",
            "has_classifier": bool  # True if classifier was used
        }
    """

    whisper_model, classifier = load_models()
    device = whisper_model.device

    # Load and preprocess audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)

    # Get Whisper transcription
    result = whisper_model.transcribe(audio_path, language="en", fp16=False)
    text = result["text"].strip()

    # If classifier is available, run analysis
    if classifier is not None:
        with torch.no_grad():
            # Get embeddings from encoder
            embeddings = whisper_model.encoder(mel.unsqueeze(0))  # (1, time, dim)

            # Run classifier
            outputs = classifier(embeddings)

            # Extract predictions
            uncertainty_score = outputs["uncertainty"].item()
            confidence_score = outputs["confidence"].item()
            emotion_logits = outputs["emotion_logits"]
            emotion_idx = emotion_logits.argmax(dim=1).item()

            emotion_map = {0: "calm", 1: "medium", 2: "heated"}
            emotional_intensity = emotion_map[emotion_idx]

            # Generate analysis
            analysis = generate_analysis(text, uncertainty_score, confidence_score, emotional_intensity)

            return {
                "text": text,
                "uncertainty_score": uncertainty_score,
                "confidence_score": confidence_score,
                "emotional_intensity": emotional_intensity,
                "analysis": analysis,
                "has_classifier": True
            }
    else:
        # Fallback: no classifier available, basic transcription
        return {
            "text": text,
            "uncertainty_score": None,
            "confidence_score": None,
            "emotional_intensity": "unknown",
            "analysis": "Classifier not available. Install trained model to models/argument_classifier.pt",
            "has_classifier": False
        }

def generate_analysis(text: str, uncertainty: float, confidence: float, emotion: str) -> str:
    """Generate human-readable analysis"""

    parts = []

    # Uncertainty analysis
    if uncertainty > 0.6:
        parts.append(f"High uncertainty detected (score: {uncertainty:.2f}). Speaker uses hedging language.")
    elif uncertainty > 0.3:
        parts.append(f"Moderate uncertainty (score: {uncertainty:.2f}). Some tentative phrasing.")
    else:
        parts.append(f"Low uncertainty (score: {uncertainty:.2f}). Speaker is direct.")

    # Confidence analysis
    if confidence > 0.6:
        parts.append(f"High confidence markers (score: {confidence:.2f}). Assertive claims.")
    elif confidence > 0.3:
        parts.append(f"Moderate confidence (score: {confidence:.2f}). Balanced assertions.")
    else:
        parts.append(f"Low confidence markers (score: {confidence:.2f}). Few strong claims.")

    # Emotional analysis
    emotion_descriptions = {
        "calm": "Calm, measured tone throughout.",
        "medium": "Moderate emotional engagement.",
        "heated": "Elevated emotional intensity detected."
    }
    parts.append(emotion_descriptions.get(emotion, "Unknown emotional state."))

    return " ".join(parts)

def analyze_argument_segment(audio_path: str, speaker_id: str) -> dict:
    """
    Analyze a single speaker's audio segment
    Useful for per-speaker analysis in arguments

    Args:
        audio_path: Path to speaker's audio segment
        speaker_id: Speaker identifier (e.g., "SPEAKER_00")

    Returns:
        dict with speaker ID and analysis
    """

    result = enhanced_transcribe(audio_path)
    result["speaker_id"] = speaker_id

    return result

# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enhanced_transcription.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    if not Path(audio_file).exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    print(f"\n🎙️  Analyzing: {audio_file}\n")

    result = enhanced_transcribe(audio_file)

    print("=" * 60)
    print("TRANSCRIPTION:")
    print("=" * 60)
    print(result["text"])
    print()

    if result["has_classifier"]:
        print("=" * 60)
        print("ARGUMENT ANALYSIS:")
        print("=" * 60)
        print(f"Uncertainty Score:     {result['uncertainty_score']:.2f}")
        print(f"Confidence Score:      {result['confidence_score']:.2f}")
        print(f"Emotional Intensity:   {result['emotional_intensity']}")
        print()
        print("ANALYSIS:")
        print(result["analysis"])
    else:
        print("\n⚠️  Classifier not available. Using basic transcription only.")
        print("   Train the classifier first: python linear_probe_approach.py")

    print("\n" + "=" * 60)
