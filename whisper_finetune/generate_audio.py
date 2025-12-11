#!/usr/bin/env python3
"""
Convert synthetic transcripts to audio using macOS 'say' command
This creates the audio files needed for Whisper fine-tuning
"""

import os
import json
import subprocess
from pathlib import Path
import random

# Input/Output
DATA_FILE = Path("training_data/synthetic_arguments.json")
AUDIO_DIR = Path("training_data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# macOS voices for different speakers
MALE_VOICES = ["Alex", "Daniel", "Fred"]
FEMALE_VOICES = ["Samantha", "Victoria", "Karen"]

def text_to_speech(text: str, output_path: str, voice: str = "Alex", rate: int = 180):
    """
    Convert text to speech using macOS 'say' command

    Args:
        text: Text to convert
        output_path: Where to save .wav file
        voice: Voice name (Alex, Samantha, etc.)
        rate: Words per minute (default 180 for natural speech)
    """
    cmd = [
        "say",
        "-v", voice,
        "-r", str(rate),
        "-o", output_path,
        "--data-format=LEI16@16000",  # 16kHz, 16-bit, little-endian (Whisper format)
        text
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ TTS failed: {e}")
        return False

def parse_and_generate_audio(entry: dict):
    """
    Parse transcript and generate audio for each speaker turn

    For training Whisper, we need:
    1. Audio file (.wav)
    2. Plain text transcription
    3. Enhanced text with markers (what we're teaching Whisper)
    """

    entry_id = entry["id"]
    raw_transcript = entry["raw_transcript"]

    # Parse speaker turns
    lines = raw_transcript.split('\n')
    speaker_turns = []

    for line in lines:
        if not line.strip() or ':' not in line:
            continue

        parts = line.split(':', 1)
        if len(parts) != 2:
            continue

        speaker = parts[0].strip()
        text = parts[1].strip()

        # Remove timestamp and emotional cues for audio generation
        if '[' in text:
            # Remove all bracketed content
            import re
            text = re.sub(r'\[.*?\]', '', text).strip()

        if text:
            speaker_turns.append({
                "speaker": speaker,
                "text": text
            })

    if len(speaker_turns) == 0:
        return None

    # Assign voices to speakers
    speakers = list(set(turn["speaker"] for turn in speaker_turns))
    voice_map = {}

    for i, speaker in enumerate(speakers):
        if i % 2 == 0:
            voice_map[speaker] = random.choice(MALE_VOICES)
        else:
            voice_map[speaker] = random.choice(FEMALE_VOICES)

    # Generate full conversation audio
    # We'll create one long audio file for the entire argument
    full_text = " ... ".join(turn["text"] for turn in speaker_turns)

    # Alternate between voices for more realistic conversation
    # For simplicity, we'll just use one voice and let Whisper learn from the text
    primary_voice = voice_map[speakers[0]]

    # Determine speaking rate based on emotional intensity
    rate = 180  # Default
    if entry["metadata"]["emotional_intensity"] == "heated":
        rate = 200  # Faster when heated

    output_path = AUDIO_DIR / f"{entry_id}.wav"

    success = text_to_speech(full_text, str(output_path), voice=primary_voice, rate=rate)

    if success:
        return {
            "audio_path": str(output_path),
            "plain_text": entry["plain_text"],
            "enhanced_text": entry["enhanced_text"],
            "metadata": entry["metadata"]
        }
    else:
        return None

def generate_all_audio():
    """Generate audio for all synthetic transcripts"""

    if not DATA_FILE.exists():
        print(f"❌ Dataset not found: {DATA_FILE}")
        print("   Run generate_training_data.py first!")
        return

    # Load dataset
    with open(DATA_FILE) as f:
        dataset = json.load(f)

    print(f"🎙️  Generating audio for {len(dataset)} transcripts...")
    print(f"   Output: {AUDIO_DIR}/")

    training_data = []
    failed = 0

    for i, entry in enumerate(dataset):
        entry_id = entry["id"]
        print(f"\n[{i+1}/{len(dataset)}] {entry_id}: {entry['topic'][:40]}...")

        result = parse_and_generate_audio(entry)

        if result:
            training_data.append({
                "id": entry_id,
                "audio_path": result["audio_path"],
                "plain_text": result["plain_text"],
                "enhanced_text": result["enhanced_text"],
                "metadata": result["metadata"]
            })
            print(f"  ✅ Audio generated")
        else:
            failed += 1
            print(f"  ❌ Failed")

    # Save training manifest
    manifest_file = Path("training_data/training_manifest.json")
    with open(manifest_file, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"\n✅ Audio generation complete!")
    print(f"   Success: {len(training_data)}/{len(dataset)}")
    print(f"   Failed: {failed}")
    print(f"   Manifest: {manifest_file}")

if __name__ == "__main__":
    generate_all_audio()
