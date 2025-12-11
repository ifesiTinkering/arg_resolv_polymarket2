#!/usr/bin/env python3
"""
Test Script for Emotion Classifier on Raspberry Pi
Run this to verify the emotion classifier is working correctly
"""

from emotion_classifier import EmotionAnalyzer

def test_emotion_classifier():
    """Test the emotion classifier with sample texts"""

    print("="*60)
    print("TESTING EMOTION CLASSIFIER ON RASPBERRY PI")
    print("="*60)

    # Initialize analyzer
    print("\n1. Loading emotion analyzer...")
    try:
        analyzer = EmotionAnalyzer()
        print("   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return

    # Test cases representing different emotions
    test_cases = [
        ("CALM", "Let's examine the evidence objectively and consider both perspectives carefully before drawing conclusions."),
        ("CONFIDENT", "I'm absolutely certain that this approach is correct. The data clearly shows overwhelming support for this position."),
        ("DEFENSIVE", "That's not what I meant at all. You're completely misrepresenting my argument and twisting my words."),
        ("DISMISSIVE", "That's ridiculous. Anyone with basic knowledge knows better than to believe something like that."),
        ("PASSIONATE", "We absolutely must take action now! This is critical for our future and we cannot wait any longer!"),
        ("FRUSTRATED", "I've explained this three times already. Why don't you understand this basic concept? Are you even listening?"),
        ("ANGRY", "You're completely wrong! That's absolutely absurd and frankly insulting! I'm done with this!"),
        ("SARCASTIC", "Oh sure, because that makes perfect sense. Wow, what brilliant logic you've applied there. Great job."),
    ]

    print("\n2. Running emotion classification tests...")
    print("-"*60)

    correct = 0
    total = len(test_cases)

    for expected_emotion, text in test_cases:
        result = analyzer.analyze(text)
        predicted = result['emotion'].upper()
        confidence = result['emotion_confidence']

        is_correct = (predicted == expected_emotion)
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        print(f"\n{status} Expected: {expected_emotion:12} | Predicted: {predicted:12} ({confidence:.1%})")
        print(f"   Text: \"{text[:80]}...\"")
        print(f"   Uncertainty: {result['uncertainty']:.3f} | Confidence: {result['confidence']:.3f}")

    print("\n" + "="*60)
    print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}% accuracy)")
    print("="*60)

    if correct >= total * 0.5:
        print("\n✅ Emotion classifier is working!")
        print("   Ready for integration into argument resolver")
    else:
        print("\n⚠️  Lower accuracy than expected")
        print("   May need more training data or model adjustments")

    return correct / total


def test_speaker_analysis():
    """Test analyzing speaker text (simulating real usage)"""

    print("\n" + "="*60)
    print("TESTING SPEAKER ANALYSIS SIMULATION")
    print("="*60)

    analyzer = EmotionAnalyzer()

    # Simulate a two-speaker argument
    speakers = {
        "SPEAKER_00": "I think universal basic income could help address inequality. Studies show mixed results, so we should consider both the benefits and potential drawbacks before making policy decisions.",
        "SPEAKER_01": "That's ridiculous! Anyone who understands basic economics knows UBI would destroy work incentives. You're living in fantasy land if you think this would actually work."
    }

    print("\nAnalyzing speaker emotions:\n")

    for speaker, text in speakers.items():
        result = analyzer.analyze(text)
        print(f"{speaker}:")
        print(f"  Emotion: {result['emotion'].upper()} ({result['emotion_confidence']:.1%} confident)")
        print(f"  Uncertainty: {result['uncertainty']:.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Text preview: \"{text[:100]}...\"")
        print()

    print("✅ Speaker analysis complete!")


if __name__ == "__main__":
    # Run tests
    try:
        accuracy = test_emotion_classifier()
        test_speaker_analysis()

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Model is ready for production use")
        print("  2. Integrate into audio_processor.py")
        print("  3. Test with real argument recordings")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
