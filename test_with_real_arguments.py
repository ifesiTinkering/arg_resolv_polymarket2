#!/usr/bin/env python3
"""
Test emotion classifier with real argument recordings from arguments_db
"""

import json
import os
from pathlib import Path
from emotion_classifier import EmotionAnalyzer

def test_with_stored_arguments():
    """Test emotion classifier with stored arguments"""

    print("="*70)
    print("TESTING EMOTION CLASSIFIER WITH REAL ARGUMENTS")
    print("="*70)

    # Initialize analyzer
    print("\n1. Loading emotion analyzer...")
    try:
        analyzer = EmotionAnalyzer()
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return

    # Find arguments database
    db_path = Path("arguments_db")
    if not db_path.exists():
        print(f"\n❌ arguments_db not found at {db_path}")
        return

    # Load all argument JSON files
    argument_files = sorted(db_path.glob("*/argument.json"))

    if not argument_files:
        print(f"\n❌ No arguments found in {db_path}")
        return

    print(f"\n2. Found {len(argument_files)} stored arguments")
    print("-"*70)

    # Analyze each argument
    for i, arg_file in enumerate(argument_files, 1):
        try:
            # Load argument data
            with open(arg_file) as f:
                arg_data = json.load(f)

            print(f"\n{'='*70}")
            print(f"ARGUMENT #{i}: {arg_data.get('title', 'Untitled')}")
            print(f"ID: {arg_data.get('id', 'unknown')}")
            print(f"Date: {arg_data.get('timestamp', 'unknown')}")
            print(f"{'='*70}")

            # Analyze each speaker
            speakers = arg_data.get('speakers', {})

            if not speakers:
                print("   ⚠️  No speaker data found")
                continue

            print(f"\n📊 Speaker Emotion Analysis:")
            print("-"*70)

            for speaker_id, speaker_data in speakers.items():
                transcript = speaker_data.get('transcript', '')

                if not transcript or len(transcript.strip()) < 10:
                    print(f"\n{speaker_id}:")
                    print("   ⚠️  Transcript too short for analysis")
                    continue

                # Analyze emotion
                emotion_result = analyzer.analyze(transcript)

                print(f"\n{speaker_id}:")
                print(f"   Emotion: {emotion_result['emotion'].upper()}")
                print(f"   Confidence: {emotion_result['emotion_confidence']:.1%}")
                print(f"   Uncertainty: {emotion_result['uncertainty']:.3f}")
                print(f"   Speaker Confidence: {emotion_result['confidence']:.3f}")
                print(f"   Word Count: {speaker_data.get('word_count', 0)}")
                print(f"   Transcript Preview: \"{transcript[:150]}...\"")

            # Show verdict preview
            verdict = arg_data.get('verdict', '')
            if verdict:
                print(f"\n📋 Verdict Preview:")
                print(f"   {verdict.split(chr(10))[0][:100]}...")

        except Exception as e:
            print(f"   ❌ Error processing argument: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"   Analyzed {len(argument_files)} arguments")
    print(f"{'='*70}")


def analyze_single_argument(argument_id):
    """Analyze a specific argument by ID"""

    print(f"\nAnalyzing argument: {argument_id}")

    analyzer = EmotionAnalyzer()

    arg_file = Path(f"arguments_db/{argument_id}/argument.json")

    if not arg_file.exists():
        print(f"❌ Argument not found: {arg_file}")
        return

    with open(arg_file) as f:
        arg_data = json.load(f)

    print(f"\nTitle: {arg_data.get('title', 'Untitled')}")
    print(f"Date: {arg_data.get('timestamp', 'unknown')}")
    print("\nSpeaker Emotions:")

    for speaker_id, speaker_data in arg_data.get('speakers', {}).items():
        transcript = speaker_data.get('transcript', '')
        if transcript:
            result = analyzer.analyze(transcript)
            print(f"\n{speaker_id}:")
            print(f"  {result['emotion'].upper()} ({result['emotion_confidence']:.1%})")
            print(f"  Uncertainty: {result['uncertainty']:.3f}")
            print(f"  Confidence: {result['confidence']:.3f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Analyze specific argument
        analyze_single_argument(sys.argv[1])
    else:
        # Analyze all arguments
        test_with_stored_arguments()
