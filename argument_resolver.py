import os, time, tempfile, shutil, subprocess, json, re, asyncio, requests
import gradio as gr
import whisper
import fastapi_poe as fp
from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

POE_API_KEY = os.getenv("POE_API_KEY")
if not POE_API_KEY:
    raise ValueError("POE_API_KEY not found in environment variables. Please set it in .env file")

# HuggingFace token - only needs READ access
# 1. Get token: https://huggingface.co/settings/tokens
# 2. Accept agreement: https://huggingface.co/pyannote/speaker-diarization-3.1
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables. Please set it in .env file")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def _assert_ffmpeg():
    from shutil import which
    if not which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg")
_assert_ffmpeg()

print("[INFO] Argument Resolver initializing...")
_WHISPER = None
_DIARIZATION = None

def get_whisper_model():
    global _WHISPER
    if _WHISPER is None:
        print("[INFO] Loading Whisper model (tiny.en for speed)...")
        t0 = time.time()
        # Use 'tiny.en' - faster and you already have it cached
        _WHISPER = whisper.load_model("tiny.en", device="cpu")
        print(f"[INFO] Whisper loaded in {time.time()-t0:.2f}s ✅")
    return _WHISPER

def get_diarization_pipeline():
    global _DIARIZATION
    if _DIARIZATION is None:
        print("[INFO] Loading speaker diarization pipeline...")
        t0 = time.time()
        try:
            _DIARIZATION = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=HUGGINGFACE_TOKEN
            )
            _DIARIZATION.to(torch.device("cpu"))
            print(f"[INFO] Diarization loaded in {time.time()-t0:.2f}s ✅")
        except Exception as e:
            print(f"[ERROR] Failed to load diarization: {e}")
            print("[INFO] Check HUGGINGFACE_TOKEN and user agreement")
            raise
    return _DIARIZATION

def _safe_copy(src_path: str) -> str:
    ext = os.path.splitext(src_path)[1] or ".wav"
    dst = tempfile.NamedTemporaryFile(suffix=ext, delete=False).name
    shutil.copy(src_path, dst)
    return dst

def _ffmpeg_to_wav16k_mono(src_path: str, max_seconds: int = 300) -> str:
    """Convert to WAV, support up to 5 minutes"""
    out_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg","-y","-i",src_path,"-t",str(max_seconds),
        "-ac","1","-ar","16000","-vn","-hide_banner","-loglevel","error",out_wav
    ]
    print("[DEBUG] Converting audio...")
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg timeout")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e}")
    print(f"[DEBUG] Audio converted in {time.time()-t0:.2f}s")
    return out_wav

def extract_speaker_segments(audio_path: str, num_speakers: int = 2) -> list:
    """Identify when different speakers are talking"""
    try:
        diarization = get_diarization_pipeline()
        print(f"[DEBUG] Running diarization (expecting {num_speakers} speakers)...")
        t0 = time.time()

        # Run diarization with specified number of speakers
        dia_result = diarization(audio_path, num_speakers=num_speakers)

        segments = []
        # New pyannote 4.x API: DiarizeOutput has .speaker_diarization attribute
        print(f"[DEBUG] Diarization result type: {type(dia_result)}")

        # Access the Annotation object from DiarizeOutput
        annotation = dia_result.speaker_diarization

        # Iterate through the annotation using itertracks
        for turn, track, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start
            })

        print(f"[DEBUG] Found {len(segments)} speech segments across {len(set(s['speaker'] for s in segments))} speakers in {time.time()-t0:.2f}s")
        return segments
    except Exception as e:
        print(f"[ERROR] Diarization failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def extract_audio_segment(audio_path: str, start_sec: float, end_sec: float) -> str:
    """Extract specific time segment from audio"""
    out_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ss", str(start_sec), "-t", str(duration),
        "-ac", "1", "-ar", "16000", "-vn",
        "-hide_banner", "-loglevel", "error", out_wav
    ]
    try:
        subprocess.run(cmd, check=True, timeout=30)
        return out_wav
    except Exception as e:
        print(f"[ERROR] Failed to extract segment: {e}")
        return None

def transcribe_segment(audio_path: str) -> str:
    """Transcribe a single audio segment"""
    model = get_whisper_model()
    result = model.transcribe(
        audio_path, language="en", fp16=False, temperature=0.0,
        condition_on_previous_text=True,  # Better for conversations
    )
    return (result.get("text") or "").strip() or ""

async def get_llm_response(prompt: str, model: str = "GPT-4o-Mini") -> str:
    """Get response from LLM via Poe"""
    message = fp.ProtocolMessage(role="user", content=prompt)
    full_response = ""
    async for partial in fp.get_bot_response(
        messages=[message],
        bot_name=model,
        api_key=POE_API_KEY
    ):
        full_response += partial.text
    return full_response

def search_polymarket(query: str) -> str:
    """Search Polymarket for relevant prediction markets"""
    try:
        # Polymarket API endpoint for active markets
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            "limit": 5,
            "closed": "false"
        }
        headers = {"Accept": "application/json"}

        response = requests.get(url, params=params, headers=headers, timeout=10)

        if response.status_code != 200:
            return f"Polymarket API error: {response.status_code}"

        markets = response.json()

        # Filter markets relevant to query
        relevant = []
        query_lower = query.lower()
        for market in markets:
            question = market.get("question", "")
            if any(word in question.lower() for word in query_lower.split()):
                relevant.append({
                    "question": question,
                    "description": market.get("description", ""),
                    "volume": market.get("volume", 0),
                    "outcomes": market.get("outcomes", [])
                })

        if not relevant:
            return "No relevant Polymarket markets found"

        # Format results
        result = "📊 Polymarket Markets:\n"
        for i, m in enumerate(relevant[:3], 1):
            result += f"\n{i}. {m['question']}\n"
            if m['outcomes']:
                result += f"   Outcomes: {', '.join(m['outcomes'])}\n"

        return result
    except Exception as e:
        return f"Polymarket search error: {e}"

async def web_search(query: str) -> str:
    """Search the web for factual information"""
    try:
        # Use LLM to search web via Poe (which has web access)
        search_prompt = f"""Search the web for factual information about: {query}

Provide a concise summary with sources. Focus on verified facts, statistics, and expert opinions."""

        result = await get_llm_response(search_prompt, model="GPT-4o-Mini")
        return result
    except Exception as e:
        return f"Web search error: {e}"

async def fact_check_claims(transcript: dict, speaker1_claims: str, speaker2_claims: str) -> str:
    """
    Fact-check both speakers' claims using Polymarket and web search
    """
    print("[DEBUG] Extracting key claims...")

    # Extract main claims from each speaker
    claim_extraction_prompt = f"""Analyze this argument transcript and extract the main factual claims made by each speaker.

Speaker 1 said: {speaker1_claims}
Speaker 2 said: {speaker2_claims}

Extract 2-3 key verifiable claims from EACH speaker. Format as:
SPEAKER 1 CLAIMS:
- Claim 1
- Claim 2

SPEAKER 2 CLAIMS:
- Claim 1
- Claim 2

Only include factual claims that can be verified, not opinions."""

    claims = await get_llm_response(claim_extraction_prompt)
    print("[DEBUG] Claims extracted")

    # Parse claims (simple approach)
    speaker1_claims_list = []
    speaker2_claims_list = []
    current_speaker = None

    for line in claims.split('\n'):
        line = line.strip()
        if 'SPEAKER 1' in line.upper():
            current_speaker = 1
        elif 'SPEAKER 2' in line.upper():
            current_speaker = 2
        elif line.startswith('-') or line.startswith('•'):
            claim = line.lstrip('-•').strip()
            if current_speaker == 1:
                speaker1_claims_list.append(claim)
            elif current_speaker == 2:
                speaker2_claims_list.append(claim)

    # Fact-check each claim
    verification_results = []

    print(f"[DEBUG] Fact-checking {len(speaker1_claims_list)} claims from Speaker 1...")
    for claim in speaker1_claims_list:
        print(f"  Checking: {claim[:50]}...")
        polymarket_data = search_polymarket(claim)
        web_data = await web_search(claim)

        verification_results.append({
            "speaker": "Speaker 1",
            "claim": claim,
            "polymarket": polymarket_data,
            "web": web_data
        })
        await asyncio.sleep(0.5)  # Rate limiting

    print(f"[DEBUG] Fact-checking {len(speaker2_claims_list)} claims from Speaker 2...")
    for claim in speaker2_claims_list:
        print(f"  Checking: {claim[:50]}...")
        polymarket_data = search_polymarket(claim)
        web_data = await web_search(claim)

        verification_results.append({
            "speaker": "Speaker 2",
            "claim": claim,
            "polymarket": polymarket_data,
            "web": web_data
        })
        await asyncio.sleep(0.5)  # Rate limiting

    # Final judgment
    print("[DEBUG] Generating final judgment...")
    judgment_prompt = f"""You are an impartial fact-checker. Based on the evidence below, determine who was more correct in this argument.

SPEAKER 1 CLAIMS:
{speaker1_claims}

SPEAKER 2 CLAIMS:
{speaker2_claims}

VERIFICATION EVIDENCE:
{json.dumps(verification_results, indent=2)}

Provide:
1. A summary of which claims were accurate/inaccurate for each speaker
2. An overall verdict on who was more correct
3. Confidence level (0-100%)
4. Key supporting evidence

Format as:
## VERDICT: [Speaker 1 / Speaker 2 / TIE]
## CONFIDENCE: [0-100]%

## ANALYSIS:
[Detailed analysis]

## EVIDENCE SUMMARY:
[Key evidence]"""

    judgment = await get_llm_response(judgment_prompt, model="GPT-4o-Mini")

    return judgment

def process_argument(audio_path: str, num_speakers: int = 2, progress=gr.Progress()):
    """Main function to process an argument recording"""
    tmp_copy = None
    tmp_wav = None
    segment_files = []

    try:
        progress(0, desc="Starting analysis...")
        print(f"[DEBUG] Received audio path: {audio_path}")

        if not audio_path or not os.path.exists(audio_path):
            return "⚠️ No audio input detected.", "", ""

        # Step 1: Audio preprocessing
        progress(0.1, desc="Converting audio...")
        tmp_copy = _safe_copy(audio_path)
        tmp_wav = _ffmpeg_to_wav16k_mono(tmp_copy, max_seconds=300)

        # Step 2: Speaker diarization
        progress(0.2, desc="Identifying speakers...")
        segments = extract_speaker_segments(tmp_wav, num_speakers=num_speakers)

        if not segments:
            return "❌ Could not identify speakers", "", ""

        # Step 3: Transcribe by speaker
        progress(0.4, desc="Transcribing conversation...")
        speaker_groups = {}
        for seg in segments:
            spk = seg["speaker"]
            if spk not in speaker_groups:
                speaker_groups[spk] = []
            speaker_groups[spk].append(seg)

        print(f"[DEBUG] Found {len(speaker_groups)} speakers")

        # Build full transcript with timestamps
        full_transcript = []
        speaker_texts = {}

        for speaker_id in sorted(speaker_groups.keys()):
            spk_segments = speaker_groups[speaker_id]
            speaker_text_parts = []

            for seg in spk_segments:
                seg_file = extract_audio_segment(tmp_wav, seg["start"], seg["end"])
                if seg_file:
                    segment_files.append(seg_file)
                    text = transcribe_segment(seg_file)
                    if text:
                        timestamp = f"[{seg['start']:.1f}s]"
                        full_transcript.append(f"{timestamp} {speaker_id}: {text}")
                        speaker_text_parts.append(text)

            speaker_texts[speaker_id] = " ".join(speaker_text_parts)

        transcript_output = "\n".join(full_transcript)

        # Step 4: Fact-checking
        progress(0.6, desc="Searching Polymarket and web...")

        speakers_list = sorted(speaker_texts.keys())
        if len(speakers_list) < 2:
            return transcript_output, "Need at least 2 speakers to judge", ""

        speaker1_text = speaker_texts[speakers_list[0]]
        speaker2_text = speaker_texts[speakers_list[1]]

        progress(0.8, desc="Fact-checking claims...")
        judgment = asyncio.run(fact_check_claims(
            speaker_texts,
            speaker1_text,
            speaker2_text
        ))

        progress(1.0, desc="Complete!")

        # Format final output
        analysis_output = f"""## FULL TRANSCRIPT
{transcript_output}

## SPEAKER SUMMARIES
**Speaker 1 ({speakers_list[0]})**: {len(speaker1_text)} chars
**Speaker 2 ({speakers_list[1]})**: {len(speaker2_text)} chars
"""

        return analysis_output, judgment, "✅ Analysis complete!"

    except Exception as e:
        msg = f"❌ {type(e).__name__}: {e}"
        print("[ERROR]", msg)
        import traceback
        traceback.print_exc()
        return msg, "Error", "Error"
    finally:
        # Cleanup
        for p in [tmp_wav, tmp_copy] + segment_files:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

# Gradio Interface
with gr.Blocks(title="Argument Resolver", theme=gr.themes.Soft()) as ui:
    gr.Markdown("""
    # 🎙️ Argument Resolver

    Record an argument between two people. The system will:
    1. Identify each speaker
    2. Transcribe what they said
    3. Search Polymarket + web for facts
    4. Determine who was actually right

    **Supports up to 5 minutes of audio**
    """)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎤 Record Argument (≤5 min, English)"
            )
            num_speakers = gr.Slider(
                minimum=2,
                maximum=4,
                value=2,
                step=1,
                label="Number of speakers"
            )
            submit_btn = gr.Button("🔍 Analyze Argument", variant="primary")

        with gr.Column():
            status_output = gr.Textbox(label="Status", lines=2)

    with gr.Row():
        transcript_output = gr.Textbox(
            label="📝 Full Transcript & Analysis",
            lines=15,
            max_lines=30
        )

    with gr.Row():
        verdict_output = gr.Textbox(
            label="⚖️ Verdict & Evidence",
            lines=20,
            max_lines=40
        )

    submit_btn.click(
        fn=process_argument,
        inputs=[audio_input, num_speakers],
        outputs=[transcript_output, verdict_output, status_output]
    )

    gr.Markdown("""
    ### Setup Instructions
    1. `pip install gradio whisper fastapi_poe requests pyannote.audio torch torchaudio`
    2. Get HuggingFace token (READ access): https://huggingface.co/settings/tokens
    3. Accept agreement: https://huggingface.co/pyannote/speaker-diarization-3.1
    4. Update `HUGGINGFACE_TOKEN` in the code

    ### Tips
    - Record in a quiet environment
    - Speakers should speak clearly
    - Works best with 2 speakers, supports up to 4
    - Longer arguments = better analysis
    """)

def main():
    import socket
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse
    import uvicorn
    import threading

    def _pick_free_port(prefer=7862):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", prefer))
                return prefer
        except OSError:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                return s.getsockname()[1]

    port = _pick_free_port(7862)

    # Get local IP for ESP32 to connect to
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except:
        local_ip = "127.0.0.1"
    finally:
        s.close()

    print(f"\n{'='*60}")
    print(f"🎙️  ARGUMENT RESOLVER")
    print(f"{'='*60}")
    print(f"Web UI: http://127.0.0.1:{port}")
    print(f"ESP32 upload to: http://{local_ip}:{port}/upload")
    print(f"\n⚠️  Before first use:")
    print(f"1. pip install pyannote.audio torch torchaudio")
    print(f"2. Set HUGGINGFACE_TOKEN in code (READ access only)")
    print(f"3. Accept user agreement at pyannote/speaker-diarization-3.1")
    print(f"\n📡 ESP32 Setup:")
    print(f"Update LAPTOP_IP in esp32_audio_recorder.py to: {local_ip}")
    print(f"{'='*60}\n")

    # Create FastAPI app for ESP32 upload endpoint
    app = FastAPI()

    @app.post("/upload")
    async def upload_audio(file: UploadFile = File(...)):
        """Endpoint for ESP32 to upload audio files"""
        print(f"\n[ESP32] Received audio upload: {file.filename}")

        try:
            # Save uploaded file temporarily
            temp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)

            print(f"[ESP32] Saved to {temp_path}, processing...")

            # Process using the same function
            transcript, verdict, status = process_argument(temp_path, num_speakers=2)

            # Extract verdict summary
            verdict_lines = verdict.split('\n')
            verdict_summary = "Unknown"
            for line in verdict_lines:
                if line.startswith("## VERDICT:"):
                    verdict_summary = line.replace("## VERDICT:", "").strip()
                    break

            print(f"[ESP32] Result: {verdict_summary}")

            return JSONResponse({
                "success": True,
                "verdict": verdict_summary,
                "transcript": transcript[:200],  # First 200 chars
                "full_verdict": verdict
            })

        except Exception as e:
            print(f"[ESP32] Error: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse({
                "success": False,
                "error": str(e)
            }, status_code=500)
        finally:
            # Cleanup
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass

    # Mount Gradio app to FastAPI
    app = gr.mount_gradio_app(app, ui, path="/")

    # Run server
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

if __name__ == "__main__":
    main()
