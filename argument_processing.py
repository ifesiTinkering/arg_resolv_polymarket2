"""
Argument processing functions - core logic without UI
Can be imported by both argument_resolver.py (Gradio UI) and pi_record_and_process.py (Pi recorder)
"""
import os

# Set torchaudio backend BEFORE any imports that use torchaudio
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

# Allow loading pretrained models from HuggingFace (trusted source)
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

import time, tempfile, shutil, subprocess, json, re, asyncio, requests
import whisper
import fastapi_poe as fp
import torch

# Monkey patch torch.load to disable weights_only for HuggingFace models
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

POE_API_KEY = os.getenv("POE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def _assert_ffmpeg():
    from shutil import which
    if not which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")
_assert_ffmpeg()

print("[INFO] Argument processing module initializing...")
_WHISPER = None
_DIARIZATION = None

def get_whisper_model():
    global _WHISPER
    if _WHISPER is None:
        print("[INFO] Loading Whisper model (tiny.en for speed)...")
        t0 = time.time()
        _WHISPER = whisper.load_model("tiny.en", device="cpu")
        print(f"[INFO] Whisper loaded in {time.time()-t0:.2f}s ✅")
    return _WHISPER

def get_diarization_pipeline():
    global _DIARIZATION
    if _DIARIZATION is None:
        print("[INFO] Loading speaker diarization pipeline...")
        t0 = time.time()
        try:
            # Try pyannote 4.x API first (token parameter)
            try:
                _DIARIZATION = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=HUGGINGFACE_TOKEN
                )
            except TypeError:
                # Fall back to pyannote 3.x API (use_auth_token parameter)
                _DIARIZATION = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HUGGINGFACE_TOKEN
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
        # Support both pyannote 3.x and 4.x APIs
        print(f"[DEBUG] Diarization result type: {type(dia_result)}")

        # pyannote 4.x returns DiarizeOutput with .speaker_diarization attribute
        # pyannote 3.x returns Annotation directly
        if hasattr(dia_result, 'speaker_diarization'):
            annotation = dia_result.speaker_diarization
        else:
            annotation = dia_result

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
