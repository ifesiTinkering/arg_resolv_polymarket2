#!/usr/bin/env python3
"""
Generate synthetic argument training data for Whisper fine-tuning
This creates audio + enhanced transcripts with uncertainty markers and emotional cues
"""

import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import fastapi_poe as fp

load_dotenv()

POE_API_KEY = os.getenv("POE_API_KEY")

# Output directory
OUTPUT_DIR = Path("training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Uncertainty markers to train model to detect
UNCERTAINTY_MARKERS = [
    "I think", "I believe", "maybe", "probably", "possibly",
    "I'm not sure", "it seems like", "I guess", "presumably",
    "in my opinion", "I suppose", "might be", "could be"
]

# Confidence markers (opposite)
CONFIDENCE_MARKERS = [
    "I know", "definitely", "certainly", "obviously", "clearly",
    "without a doubt", "absolutely", "I'm certain", "it's a fact",
    "undeniably", "for sure", "guaranteed"
]

# Argument topics
ARGUMENT_TOPICS = [
    "climate change policies",
    "artificial intelligence regulation",
    "universal basic income",
    "space exploration funding",
    "remote work vs office work",
    "electric vehicles mandate",
    "cryptocurrency as currency",
    "social media age restrictions",
    "nuclear energy expansion",
    "genetic engineering ethics",
    "privacy vs security debate",
    "college education value",
    "meat consumption and environment",
    "automation and jobs",
    "healthcare system design",
    "renewable energy transition",
    "urban planning and cars",
    "screen time for children",
    "globalization impacts",
    "tax policy fairness"
]

async def generate_argument_transcript(topic: str, style: str = "balanced") -> dict:
    """
    Generate a realistic argument transcript between two people

    Args:
        topic: What they're arguing about
        style: "uncertain" (lots of hedging), "confident" (assertive), or "balanced"

    Returns:
        dict with transcript and metadata
    """

    style_instructions = {
        "uncertain": f"Use LOTS of uncertainty markers like: {', '.join(UNCERTAINTY_MARKERS[:5])}. Make speakers sound hesitant and unsure.",
        "confident": f"Use LOTS of confidence markers like: {', '.join(CONFIDENCE_MARKERS[:5])}. Make speakers sound very assertive and certain.",
        "balanced": "Mix both confident and uncertain statements naturally, as people actually talk in arguments."
    }

    prompt = f"""Generate a realistic 30-second argument transcript between two people about: {topic}

Requirements:
- Exactly 2 speakers (SPEAKER_1 and SPEAKER_2)
- Each speaker should speak 3-4 times
- Keep it natural and conversational (like real people arguing)
- {style_instructions[style]}
- Include interruptions occasionally (mark with [interrupts])
- Include emotional cues like [raises voice] or [calmly] when relevant
- Total length: ~150-250 words (30 seconds of speech)

Format as:
SPEAKER_1: [timestamp] [emotional cue if any] Text here
SPEAKER_2: [timestamp] [emotional cue if any] Text here

Example format:
SPEAKER_1: [0.0s] I think climate change is the biggest threat we face today.
SPEAKER_2: [4.2s] [interrupts] That's not true! The data clearly shows...
SPEAKER_1: [8.5s] [raises voice] Let me finish! I'm saying that maybe we should...

Start now:"""

    try:
        message = fp.ProtocolMessage(role="user", content=prompt)
        full_response = ""
        async for partial in fp.get_bot_response(
            messages=[message],
            bot_name="GPT-4o-Mini",
            api_key=POE_API_KEY
        ):
            full_response += partial.text

        return {
            "topic": topic,
            "style": style,
            "transcript": full_response.strip(),
            "has_uncertainty": style in ["uncertain", "balanced"],
            "has_confidence": style in ["confident", "balanced"]
        }
    except Exception as e:
        print(f"Error generating transcript: {e}")
        return None

async def create_enhanced_transcript(raw_transcript: str) -> dict:
    """
    Parse transcript and create enhanced version with markers

    Returns:
        {
            "plain_text": "standard transcription",
            "enhanced_text": "transcription [UNCERTAIN] with [CONFIDENT] markers [HEATED]",
            "metadata": {
                "uncertainty_count": 3,
                "confidence_count": 2,
                "emotional_intensity": "medium",
                "interruption_count": 1
            }
        }
    """

    lines = raw_transcript.split('\n')
    plain_parts = []
    enhanced_parts = []

    uncertainty_count = 0
    confidence_count = 0
    emotional_intensity = "calm"
    interruption_count = 0

    for line in lines:
        if not line.strip() or ':' not in line:
            continue

        # Extract speaker and text
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue

        speaker = parts[0].strip()
        text = parts[1].strip()

        # Remove timestamp if present
        if '[' in text and 's]' in text:
            text = text.split('s]', 1)[1].strip()

        # Check for emotional cues
        emotional_cue = None
        if '[' in text and ']' in text:
            # Extract first bracketed expression
            start = text.index('[')
            end = text.index(']')
            emotional_cue = text[start+1:end].lower()
            text = text[:start] + text[end+1:]

            if any(word in emotional_cue for word in ['raise', 'loud', 'angry', 'heated']):
                emotional_intensity = "heated"
                interruption_count += 1 if 'interrupt' in emotional_cue else 0

        # Plain version (just the text)
        plain_parts.append(text.strip())

        # Enhanced version with markers
        enhanced = text.strip()

        # Mark uncertainty
        for marker in UNCERTAINTY_MARKERS:
            if marker.lower() in enhanced.lower():
                # Add marker after the phrase
                enhanced = enhanced.replace(marker, f"{marker} [UNCERTAIN]")
                uncertainty_count += 1
                break

        # Mark confidence
        for marker in CONFIDENCE_MARKERS:
            if marker.lower() in enhanced.lower():
                enhanced = enhanced.replace(marker, f"{marker} [CONFIDENT]")
                confidence_count += 1
                break

        # Add emotional marker
        if emotional_cue and ('raise' in emotional_cue or 'loud' in emotional_cue):
            enhanced += " [HEATED]"

        enhanced_parts.append(enhanced)

    return {
        "plain_text": " ".join(plain_parts),
        "enhanced_text": " ".join(enhanced_parts),
        "metadata": {
            "uncertainty_count": uncertainty_count,
            "confidence_count": confidence_count,
            "emotional_intensity": emotional_intensity,
            "interruption_count": interruption_count
        }
    }

async def generate_dataset(num_samples: int = 100):
    """
    Generate training dataset

    Args:
        num_samples: Number of argument transcripts to generate
    """

    print(f"🎯 Generating {num_samples} synthetic argument transcripts...")
    print(f"   Output: {OUTPUT_DIR}/")

    dataset = []

    for i in range(num_samples):
        # Pick random topic and style
        topic = random.choice(ARGUMENT_TOPICS)
        style = random.choice(["uncertain", "confident", "balanced", "balanced"])  # More balanced

        print(f"\n[{i+1}/{num_samples}] Generating: {topic} ({style})...")

        # Generate transcript
        result = await generate_argument_transcript(topic, style)

        if not result:
            print(f"  ❌ Failed")
            continue

        # Create enhanced version
        enhanced = await create_enhanced_transcript(result["transcript"])

        # Save entry
        entry = {
            "id": f"arg_{i:04d}",
            "topic": topic,
            "style": style,
            "raw_transcript": result["transcript"],
            "plain_text": enhanced["plain_text"],
            "enhanced_text": enhanced["enhanced_text"],
            "metadata": enhanced["metadata"]
        }

        dataset.append(entry)

        print(f"  ✅ Generated ({len(enhanced['plain_text'])} chars)")
        print(f"     Uncertainty: {enhanced['metadata']['uncertainty_count']}, "
              f"Confidence: {enhanced['metadata']['confidence_count']}, "
              f"Intensity: {enhanced['metadata']['emotional_intensity']}")

        # Rate limiting
        await asyncio.sleep(1)

    # Save dataset
    output_file = OUTPUT_DIR / "synthetic_arguments.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Dataset saved to {output_file}")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Ready for audio generation!")

    # Print statistics
    total_uncertainty = sum(e["metadata"]["uncertainty_count"] for e in dataset)
    total_confidence = sum(e["metadata"]["confidence_count"] for e in dataset)
    heated_count = sum(1 for e in dataset if e["metadata"]["emotional_intensity"] == "heated")

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total uncertainty markers: {total_uncertainty}")
    print(f"   Total confidence markers: {total_confidence}")
    print(f"   Heated arguments: {heated_count}")
    print(f"   Calm arguments: {len(dataset) - heated_count}")

if __name__ == "__main__":
    import sys

    num_samples = 100
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])

    asyncio.run(generate_dataset(num_samples))
