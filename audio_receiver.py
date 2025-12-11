#!/usr/bin/env python3
"""
Simple Audio Receiver
Receives audio files from Raspberry Pi and saves them locally
"""

import os
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# Configuration
SAVE_DIR = "/Users/dimmaonubogu/aiot_project/received_audio"
PORT = 7862

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

app = FastAPI()

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Receive audio file from Raspberry Pi and save it"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"argument_{timestamp}.wav"
    filepath = os.path.join(SAVE_DIR, filename)

    try:
        # Save the uploaded file
        print(f"\n[RECEIVED] Audio from Raspberry Pi")
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)

        file_size = os.path.getsize(filepath)
        print(f"[SAVED] {filename} ({file_size} bytes)")
        print(f"[PATH] {filepath}")

        return JSONResponse({
            "success": True,
            "message": "Audio received and saved",
            "filename": filename,
            "filepath": filepath,
            "size_bytes": file_size
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/")
async def root():
    """Status endpoint"""
    files = os.listdir(SAVE_DIR)
    audio_files = [f for f in files if f.endswith('.wav')]

    return {
        "status": "running",
        "save_directory": SAVE_DIR,
        "total_files": len(audio_files),
        "files": audio_files[-5:]  # Show last 5 files
    }

def main():
    print("=" * 60)
    print("🎙️  AUDIO RECEIVER")
    print("=" * 60)
    print(f"Listening on: http://0.0.0.0:{PORT}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"\nRaspberry Pi should send to: http://10.46.130.179:{PORT}/upload")
    print("=" * 60)
    print("\nWaiting for audio files from Raspberry Pi...\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

if __name__ == "__main__":
    main()
