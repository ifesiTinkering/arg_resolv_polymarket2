#!/usr/bin/env python3
"""
Raspberry Pi Unified Processor
Records audio, processes it locally, and sends results to laptop for browsing
"""

import os
import sys
import subprocess
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

# Set environment variable to suppress torchaudio backend warnings
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

# Import processing functions from argument_resolver
import whisper
import torch
import torchaudio
from pyannote.audio import Pipeline
import tempfile
import shutil
import json
import re
import asyncio
import fastapi_poe as fp

# Load environment variables
load_dotenv()

# ===== CONFIGURATION =====
LAPTOP_IP = os.getenv("LAPTOP_IP", "172.22.129.179")
LAPTOP_PORT = 7864  # Different port for results receiver
RECORD_DURATION = 30  # seconds
SAMPLE_RATE = 16000  # 16kHz for Whisper
AUDIO_FILE = "/tmp/argument_audio.wav"

# API Keys
POE_API_KEY = os.getenv("POE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not POE_API_KEY:
    print("Warning: POE_API_KEY not found in environment variables")
if not HUGGINGFACE_TOKEN:
    print("Warning: HUGGINGFACE_TOKEN not found in environment variables")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===== GLOBAL MODELS =====
_WHISPER = None
_DIARIZATION = None

def get_whisper_model():
    """Load Whisper model (lazy loading)"""
    global _WHISPER
    if _WHISPER is None:
        print("[INFO] Loading Whisper model (tiny.en for speed)...")
        _WHISPER = whisper.load_model("tiny.en")
        print("[INFO] Whisper model loaded")
    return _WHISPER

def get_diarization_pipeline():
    """Load speaker diarization pipeline (lazy loading)"""
    global _DIARIZATION
    if _DIARIZATION is None:
        print("[INFO] Loading speaker diarization model...")
        _DIARIZATION = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HUGGINGFACE_TOKEN
        )
        if torch.cuda.is_available():
            _DIARIZATION = _DIARIZATION.to(torch.device("cuda"))
        print("[INFO] Diarization model loaded")
    return _DIARIZATION

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

def extract_speaker_segments(audio_path):
    """Perform speaker diarization"""
    print("[INFO] Performing speaker diarization...")
    pipeline = get_diarization_pipeline()
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })

    print(f"[INFO] Found {len(set(s['speaker'] for s in segments))} speakers")
    return segments

def extract_audio_segment(input_path, output_path, start_time, end_time):
    """Extract a segment of audio using ffmpeg"""
    duration = end_time - start_time
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ss', str(start_time),
        '-t', str(duration),
        '-ar', str(SAMPLE_RATE),
        '-ac', '1',
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def transcribe_segment(audio_path):
    """Transcribe audio segment using Whisper"""
    model = get_whisper_model()
    result = model.transcribe(audio_path, language="en")
    return result['text'].strip()

def process_audio_locally():
    """Process the recorded audio locally on the Pi"""
    print("\n🔄 Processing audio locally...")

    try:
        # Extract speaker segments
        segments = extract_speaker_segments(AUDIO_FILE)

        # Group segments by speaker
        speaker_data = {}
        temp_dir = tempfile.mkdtemp()

        for i, seg in enumerate(segments):
            speaker = seg['speaker']
            if speaker not in speaker_data:
                speaker_data[speaker] = {'segments': [], 'text': []}

            # Extract and transcribe segment
            seg_path = os.path.join(temp_dir, f"seg_{i}.wav")
            extract_audio_segment(AUDIO_FILE, seg_path, seg['start'], seg['end'])
            text = transcribe_segment(seg_path)

            speaker_data[speaker]['segments'].append({
                'start': seg['start'],
                'end': seg['end'],
                'text': text
            })
            speaker_data[speaker]['text'].append(text)

        # Build full transcript
        transcript_lines = []
        for seg in segments:
            speaker = seg['speaker']
            matching_seg = next(s for s in speaker_data[speaker]['segments']
                              if s['start'] == seg['start'])
            transcript_lines.append(f"[{seg['start']:.1f}s] {speaker}: {matching_seg['text']}")

        full_transcript = "\n".join(transcript_lines)

        # Combine speaker text
        for speaker in speaker_data:
            speaker_data[speaker]['full_text'] = " ".join(speaker_data[speaker]['text'])

        # Clean up temp files
        shutil.rmtree(temp_dir)

        print("✓ Audio processed successfully")

        return {
            'success': True,
            'num_speakers': len(speaker_data),
            'speakers': speaker_data,
            'full_transcript': full_transcript,
            'audio_path': AUDIO_FILE
        }

    except Exception as e:
        print(f"✗ Processing failed: {e}")
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
            'speakers': {k: {'transcript': v['full_text']}
                        for k, v in result['speakers'].items()}
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
        # Clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

def main():
    print("="*60)
    print("🎙️  RASPBERRY PI ARGUMENT PROCESSOR")
    print("="*60)
    print(f"Record duration: {RECORD_DURATION} seconds")
    print(f"Laptop endpoint: {LAPTOP_IP}:{LAPTOP_PORT}")
    print("="*60)

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

    # Send to laptop (REQUIRED - all storage happens on laptop)
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
