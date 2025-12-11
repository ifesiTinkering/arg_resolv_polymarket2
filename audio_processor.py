#!/usr/bin/env python3
"""
Audio Processor for Argument Resolver
Receives audio from Raspberry Pi, processes it, and stores results
"""

import os
import tempfile
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# Import processing functions from argument_resolver
from argument_resolver import (
    _ffmpeg_to_wav16k_mono,
    extract_speaker_segments,
    extract_audio_segment,
    transcribe_segment,
    fact_check_claims,
    get_llm_response
)
import asyncio

# Import storage manager
from storage import ArgumentStorage

# Import emotion classifier
try:
    from emotion_classifier import EmotionAnalyzer
    EMOTION_ANALYSIS_ENABLED = True
    print("[INFO] Emotion analysis enabled")
except ImportError:
    EMOTION_ANALYSIS_ENABLED = False
    print("[INFO] Emotion analysis disabled (emotion_classifier.py not found)")

# Configuration
PORT = 7862
storage = ArgumentStorage()

# Initialize emotion analyzer (lazy loading)
emotion_analyzer = None

app = FastAPI()

async def generate_argument_title(transcript: str, verdict: str) -> str:
    """
    Use AI to generate a descriptive title for the argument based on content
    """
    try:
        prompt = f"""Based on this argument transcript and verdict, generate a short, descriptive title (max 60 characters).
The title should capture the main topic being debated.

Transcript preview:
{transcript[:500]}

Verdict:
{verdict[:200]}

Generate ONLY the title text, nothing else. Make it descriptive and search-friendly.
Examples of good titles:
- "AI Job Displacement by 2030"
- "Climate Change Policy Debate"
- "Benefits of Remote Work"
- "Cryptocurrency vs Traditional Banking"

Your title:"""

        title = await get_llm_response(prompt)

        # Clean up the title
        title = title.strip().strip('"').strip("'")

        # Limit length
        if len(title) > 60:
            title = title[:57] + "..."

        # If title generation failed, use a fallback
        if not title or len(title) < 5:
            # Extract first meaningful words from transcript
            words = transcript.split()[:8]
            title = " ".join(words)
            if len(title) > 60:
                title = title[:57] + "..."

        print(f"[TITLE] Generated: {title}")
        return title

    except Exception as e:
        print(f"[ERROR] Title generation failed: {e}")
        # Fallback to first words of transcript
        words = transcript.split()[:8]
        return " ".join(words)[:60]

@app.post("/upload")
async def upload_and_process_audio(file: UploadFile = File(...)):
    """
    Receive audio file from Raspberry Pi, process it, and store results
    """
    temp_audio = None
    temp_wav = None
    segment_files = []

    try:
        print(f"\n{'='*60}")
        print(f"[RECEIVED] Audio from Raspberry Pi: {file.filename}")
        print(f"{'='*60}")

        # Save uploaded file temporarily
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with open(temp_audio, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size = os.path.getsize(temp_audio)
        print(f"[INFO] File size: {file_size} bytes")

        # Step 1: Convert audio to standard format
        print("[STEP 1/5] Converting audio format...")
        temp_wav = _ffmpeg_to_wav16k_mono(temp_audio, max_seconds=300)

        # Step 2: Speaker diarization
        print("[STEP 2/5] Identifying speakers...")
        segments = extract_speaker_segments(temp_wav, num_speakers=2)

        if not segments:
            return JSONResponse({
                "success": False,
                "error": "Could not identify speakers"
            }, status_code=400)

        # Step 3: Transcribe by speaker
        print("[STEP 3/5] Transcribing conversation...")
        speaker_groups = {}
        for seg in segments:
            spk = seg["speaker"]
            if spk not in speaker_groups:
                speaker_groups[spk] = []
            speaker_groups[spk].append(seg)

        # Build full transcript
        full_transcript = []
        speaker_texts = {}

        for speaker_id in sorted(speaker_groups.keys()):
            spk_segments = speaker_groups[speaker_id]
            speaker_text_parts = []

            for seg in spk_segments:
                seg_file = extract_audio_segment(temp_wav, seg["start"], seg["end"])
                if seg_file:
                    segment_files.append(seg_file)
                    text = transcribe_segment(seg_file)
                    if text:
                        timestamp = f"[{seg['start']:.1f}s]"
                        full_transcript.append(f"{timestamp} {speaker_id}: {text}")
                        speaker_text_parts.append(text)

            speaker_texts[speaker_id] = " ".join(speaker_text_parts)

        transcript_output = "\n".join(full_transcript)
        print(f"[INFO] Transcribed {len(speaker_groups)} speakers")

        # Step 4: Fact-checking
        print("[STEP 4/5] Fact-checking claims...")
        speakers_list = sorted(speaker_texts.keys())

        if len(speakers_list) < 2:
            verdict = "Need at least 2 speakers to judge"
        else:
            speaker1_text = speaker_texts[speakers_list[0]]
            speaker2_text = speaker_texts[speakers_list[1]]

            verdict = await asyncio.create_task(fact_check_claims(
                speaker_texts,
                speaker1_text,
                speaker2_text
            ))

        # Step 5: Generate intelligent title
        print("[STEP 5/7] Generating title...")
        title = await generate_argument_title(transcript_output, verdict)

        # Step 6: Analyze emotions (moved below)
        # Step 7: Save to storage
        print("[STEP 7/7] Saving to database...")

        # Calculate audio duration
        if segments:
            duration = max(seg["end"] for seg in segments)
        else:
            duration = 0

        # Prepare speaker data with emotion analysis
        print("[STEP 6/7] Analyzing speaker emotions...")
        speakers_data = {}

        # Lazy load emotion analyzer
        global emotion_analyzer
        if EMOTION_ANALYSIS_ENABLED and emotion_analyzer is None:
            try:
                emotion_analyzer = EmotionAnalyzer()
            except Exception as e:
                print(f"[WARNING] Failed to load emotion analyzer: {e}")
                EMOTION_ANALYSIS_ENABLED = False

        for speaker_id, text in speaker_texts.items():
            speaker_data = {
                "transcript": text,
                "word_count": len(text.split())
            }

            # Add emotion analysis if available
            if EMOTION_ANALYSIS_ENABLED and emotion_analyzer:
                try:
                    emotion_result = emotion_analyzer.analyze(text)
                    speaker_data["emotion"] = emotion_result["emotion"]
                    speaker_data["emotion_confidence"] = emotion_result["emotion_confidence"]
                    speaker_data["uncertainty"] = emotion_result["uncertainty"]
                    speaker_data["confidence"] = emotion_result["confidence"]
                    print(f"  {speaker_id}: {emotion_result['emotion'].upper()} ({emotion_result['emotion_confidence']:.1%} confident)")
                except Exception as e:
                    print(f"[WARNING] Emotion analysis failed for {speaker_id}: {e}")
                    speaker_data["emotion"] = "unknown"
                    speaker_data["emotion_confidence"] = 0.0

            speakers_data[speaker_id] = speaker_data

        # Save to storage
        argument_id = storage.save_argument(
            audio_path=temp_wav,
            transcript=transcript_output,
            verdict=verdict,
            speakers=speakers_data,
            metadata={
                "duration": duration,
                "num_speakers": len(speaker_groups)
            },
            title=title
        )

        print(f"[SUCCESS] Argument saved with ID: {argument_id}")
        print(f"{'='*60}\n")

        # Return success response to Pi
        return JSONResponse({
            "success": True,
            "message": "Audio received and processed successfully",
            "argument_id": argument_id,
            "title": title,
            "timestamp": datetime.now().isoformat(),
            "num_speakers": len(speaker_groups),
            "verdict_preview": verdict.split('\n')[0] if verdict else "Processing complete"
        })

    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()

        return JSONResponse({
            "success": False,
            "error": error_msg
        }, status_code=500)

    finally:
        # Cleanup temporary files (but keep the one saved in storage)
        for p in [temp_audio] + segment_files:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

@app.get("/")
async def root():
    """Status endpoint"""
    stats = storage.get_stats()

    return {
        "status": "running",
        "service": "Argument Resolver Audio Processor",
        "database_stats": stats
    }

@app.get("/arguments")
async def list_arguments():
    """List all stored arguments"""
    arguments = storage.list_arguments(limit=50)
    return {
        "total": len(arguments),
        "arguments": arguments
    }

@app.get("/arguments/{argument_id}")
async def get_argument(argument_id: str):
    """Get specific argument details"""
    arg = storage.get_argument(argument_id)

    if not arg:
        return JSONResponse({
            "error": "Argument not found"
        }, status_code=404)

    return arg

def main():
    import socket

    # Get local IP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except:
        local_ip = "127.0.0.1"
    finally:
        s.close()

    print(f"\n{'='*60}")
    print(f"🎙️  ARGUMENT RESOLVER - AUDIO PROCESSOR")
    print(f"{'='*60}")
    print(f"Local access: http://127.0.0.1:{PORT}")
    print(f"Raspberry Pi should upload to: http://{local_ip}:{PORT}/upload")
    print(f"\nDatabase: {storage.base_dir}")
    print(f"Stats: {storage.get_stats()}")
    print(f"{'='*60}")
    print("\n⏳ Waiting for audio from Raspberry Pi...\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

if __name__ == "__main__":
    main()
