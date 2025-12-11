#!/usr/bin/env python3
"""
Raspberry Pi: Record audio and process using argument_resolver.py functions
Sends results to laptop for browsing
"""

import os

# CRITICAL: Set torchaudio backend BEFORE any imports
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

import sys
import subprocess
import requests
import time
import tempfile
import shutil
import json
from datetime import datetime
from dotenv import load_dotenv

# Import all the working processing functions from argument_processing module
from argument_processing import (
    extract_speaker_segments,
    extract_audio_segment,
    transcribe_segment,
    fact_check_claims,
    _ffmpeg_to_wav16k_mono
)
import asyncio

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====
LAPTOP_IP = os.getenv("LAPTOP_IP", "172.22.129.179")
LAPTOP_PORT = 7864  # Port for results_receiver.py
RECORD_DURATION = 30  # seconds
SAMPLE_RATE = 16000  # 16kHz for Whisper
AUDIO_FILE = "/tmp/argument_audio.wav"

print("="*60)
print("🎙️  RASPBERRY PI ARGUMENT PROCESSOR")
print("="*60)
print(f"Record duration: {RECORD_DURATION} seconds")
print(f"Laptop endpoint: {LAPTOP_IP}:{LAPTOP_PORT}")
print("="*60)

def check_microphone():
    """Check if USB microphone is detected"""
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        if 'card' in result.stdout:
            print("✓ Microphone detected")
            return True
        else:
            print("✗ No microphone detected. Please plug in USB microphone.")
            return False
    except Exception as e:
        print(f"Error checking microphone: {e}")
        return False

def record_audio(duration=RECORD_DURATION):
    """Record audio using arecord"""
    print(f"\n🎙️  Recording {duration} seconds of audio...")
    print("Speak into the microphone now!")

    cmd = [
        'arecord',
        '-D', 'plughw:2,0',  # USB microphone card 2
        '-f', 'S16_LE',      # 16-bit little-endian
        '-c', '1',            # Mono
        '-r', str(SAMPLE_RATE),  # Sample rate
        '-d', str(duration),  # Duration
        AUDIO_FILE
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Recording complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Recording failed: {e}")
        return False

def process_audio_locally():
    """Process the recorded audio using argument_resolver.py functions"""
    print("\n🔄 Processing audio locally...")

    try:
        # Step 1: Convert audio to proper format
        print("[INFO] Converting audio...")
        converted_wav = _ffmpeg_to_wav16k_mono(AUDIO_FILE, max_seconds=300)

        # Step 2: Extract speaker segments
        print("[INFO] Identifying speakers...")
        segments = extract_speaker_segments(converted_wav, num_speakers=2)

        if not segments:
            return {'success': False, 'error': 'Could not identify speakers'}

        # Step 3: Group segments by speaker and transcribe
        print("[INFO] Transcribing conversation...")
        speaker_groups = {}
        for seg in segments:
            spk = seg["speaker"]
            if spk not in speaker_groups:
                speaker_groups[spk] = []
            speaker_groups[spk].append(seg)

        # Build full transcript with timestamps
        full_transcript = []
        speaker_texts = {}

        for speaker_id in sorted(speaker_groups.keys()):
            spk_segments = speaker_groups[speaker_id]
            speaker_text_parts = []

            for seg in spk_segments:
                seg_file = extract_audio_segment(converted_wav, seg["start"], seg["end"])
                if seg_file:
                    text = transcribe_segment(seg_file)
                    if text:
                        timestamp = f"[{seg['start']:.1f}s]"
                        full_transcript.append(f"{timestamp} {speaker_id}: {text}")
                        speaker_text_parts.append(text)
                    # Cleanup segment file
                    try:
                        os.remove(seg_file)
                    except:
                        pass

            speaker_texts[speaker_id] = " ".join(speaker_text_parts)

        transcript_output = "\n".join(full_transcript)

        # Step 4: Fact-check claims (if we have at least 2 speakers)
        verdict = None
        speakers_list = sorted(speaker_texts.keys())
        if len(speakers_list) >= 2:
            print("[INFO] Fact-checking claims...")
            speaker1_text = speaker_texts[speakers_list[0]]
            speaker2_text = speaker_texts[speakers_list[1]]

            try:
                verdict = asyncio.run(fact_check_claims(
                    speaker_texts,
                    speaker1_text,
                    speaker2_text
                ))
            except Exception as e:
                print(f"[WARNING] Fact-checking failed: {e}")
                verdict = None

        # Cleanup converted wav
        try:
            os.remove(converted_wav)
        except:
            pass

        print("✓ Audio processed successfully")

        return {
            'success': True,
            'num_speakers': len(speaker_texts),
            'speakers': {k: {'transcript': v} for k, v in speaker_texts.items()},
            'full_transcript': transcript_output,
            'verdict': verdict
        }

    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def send_results_to_laptop(result, argument_id):
    """Send processing results to laptop for storage and browsing"""
    url = f"http://{LAPTOP_IP}:{LAPTOP_PORT}/receive_results"

    print(f"\n📤 Sending results to laptop...")

    # Create temporary files for sending
    temp_dir = tempfile.mkdtemp()

    try:
        # Create temporary transcript file
        transcript_path = os.path.join(temp_dir, 'transcript.txt')
        with open(transcript_path, 'w') as f:
            f.write(result['full_transcript'])

        # Create temporary metadata file
        metadata = {
            'id': argument_id,
            'timestamp': datetime.now().isoformat(),
            'num_speakers': result['num_speakers'],
            'speakers': result['speakers'],
            'full_verdict_text': result.get('verdict', '')
        }

        metadata_path = os.path.join(temp_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Prepare files to send
        files = {
            'audio': open(AUDIO_FILE, 'rb'),
            'transcript': open(transcript_path, 'rb'),
            'metadata': open(metadata_path, 'rb')
        }

        data = {'argument_id': argument_id}

        response = requests.post(url, files=files, data=data, timeout=60)

        # Close files
        for f in files.values():
            f.close()

        # Clean up temp files
        shutil.rmtree(temp_dir)

        if response.status_code == 200:
            print("✓ Results sent to laptop successfully")
            print(f"  View at: http://{LAPTOP_IP}:7863")
            return True
        else:
            print(f"✗ Failed to send results (status {response.status_code})")
            return False

    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to laptop at {LAPTOP_IP}:{LAPTOP_PORT}")
        print("  Make sure results_receiver.py is running on your laptop!")
        # Clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False
    except Exception as e:
        print(f"✗ Error sending results: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

def main():
    # Check microphone
    if not check_microphone():
        sys.exit(1)

    # Wait for user
    input("\nPress ENTER to start recording...")

    # Record
    if not record_audio():
        sys.exit(1)

    # Process locally
    result = process_audio_locally()

    if not result['success']:
        print("\n❌ Failed to process audio.")
        sys.exit(1)

    # Generate argument ID
    argument_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Send to laptop
    success = send_results_to_laptop(result, argument_id)

    if success:
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE")
        print("="*60)
        print(f"Argument ID: {argument_id}")
        print(f"Speakers detected: {result['num_speakers']}")
        print(f"View results: http://{LAPTOP_IP}:7863")
        print("="*60)
    else:
        print("\n❌ Failed to send results to laptop.")
        print("   Make sure results_receiver.py is running on your laptop!")
        sys.exit(1)

if __name__ == "__main__":
    main()
