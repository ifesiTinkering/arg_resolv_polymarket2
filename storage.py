#!/usr/bin/env python3
"""
Storage Manager for Argument Resolver
Handles saving and retrieving resolved arguments from filesystem
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional

class ArgumentStorage:
    def __init__(self, base_dir: str = "/Users/dimmaonubogu/aiot_project/arguments_db"):
        self.base_dir = base_dir
        self.arguments_dir = os.path.join(base_dir, "arguments")
        self.index_file = os.path.join(base_dir, "arguments.json")

        # Create directories if they don't exist
        os.makedirs(self.arguments_dir, exist_ok=True)

        # Initialize index file if it doesn't exist
        if not os.path.exists(self.index_file):
            self._save_index([])

    def _save_index(self, index: List[Dict]):
        """Save the index file"""
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)

    def _load_index(self) -> List[Dict]:
        """Load the index file"""
        try:
            with open(self.index_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []

    def save_argument(self,
                     audio_path: str,
                     transcript: str,
                     verdict: str,
                     speakers: Dict,
                     metadata: Dict,
                     title: str = None) -> str:
        """
        Save a resolved argument to the database

        Args:
            audio_path: Path to the audio file
            transcript: Full transcript text
            verdict: Verdict text from LLM
            speakers: Dict of speaker data
            metadata: Additional metadata (duration, num_speakers, etc.)
            title: Human-readable title for the argument (optional)

        Returns:
            argument_id: Unique ID for this argument
        """
        # Generate unique ID from timestamp
        timestamp = datetime.now()
        argument_id = timestamp.strftime("%Y%m%d_%H%M%S")

        # Create directory for this argument
        arg_dir = os.path.join(self.arguments_dir, argument_id)
        os.makedirs(arg_dir, exist_ok=True)

        # Copy audio file
        audio_dest = os.path.join(arg_dir, "audio.wav")
        shutil.copy(audio_path, audio_dest)

        # Save transcript
        transcript_path = os.path.join(arg_dir, "transcript.txt")
        with open(transcript_path, 'w') as f:
            f.write(transcript)

        # Parse verdict to extract winner and confidence
        verdict_data = self._parse_verdict(verdict)

        # Create metadata
        full_metadata = {
            "id": argument_id,
            "title": title or f"Argument {argument_id}",
            "timestamp": timestamp.isoformat(),
            "duration_seconds": metadata.get("duration", 0),
            "num_speakers": metadata.get("num_speakers", 2),
            "speakers": speakers,
            "verdict": verdict_data,
            "full_verdict_text": verdict
        }

        # Save metadata
        metadata_path = os.path.join(arg_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)

        # Update index
        index = self._load_index()
        index.append({
            "id": argument_id,
            "title": full_metadata["title"],
            "timestamp": timestamp.isoformat(),
            "winner": verdict_data.get("winner", "Unknown"),
            "num_speakers": full_metadata["num_speakers"]
        })
        self._save_index(index)

        print(f"[STORAGE] Saved argument: {argument_id}")
        print(f"[STORAGE] Title: {full_metadata['title']}")
        return argument_id

    def _parse_verdict(self, verdict_text: str) -> Dict:
        """Parse verdict text to extract structured data"""
        verdict_data = {
            "winner": "Unknown",
            "confidence": 0,
            "reasoning": verdict_text
        }

        # Try to extract winner
        lines = verdict_text.split('\n')
        for line in lines:
            if line.startswith("## VERDICT:"):
                winner_text = line.replace("## VERDICT:", "").strip()
                verdict_data["winner"] = winner_text
            elif line.startswith("## CONFIDENCE:"):
                conf_text = line.replace("## CONFIDENCE:", "").strip().replace("%", "")
                try:
                    verdict_data["confidence"] = int(conf_text)
                except ValueError:
                    pass

        return verdict_data

    def get_argument(self, argument_id: str) -> Optional[Dict]:
        """Get a specific argument by ID"""
        arg_dir = os.path.join(self.arguments_dir, argument_id)

        if not os.path.exists(arg_dir):
            return None

        # Load metadata
        metadata_path = os.path.join(arg_dir, "metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception:
            return None

        # Load transcript
        transcript_path = os.path.join(arg_dir, "transcript.txt")
        try:
            with open(transcript_path, 'r') as f:
                transcript = f.read()
        except Exception:
            transcript = ""

        # Add paths
        metadata["audio_path"] = os.path.join(arg_dir, "audio.wav")
        metadata["transcript"] = transcript

        return metadata

    def list_arguments(self, limit: Optional[int] = None) -> List[Dict]:
        """List all arguments (most recent first)"""
        index = self._load_index()

        # Sort by timestamp (newest first)
        index.sort(key=lambda x: x["timestamp"], reverse=True)

        if limit:
            index = index[:limit]

        return index

    def search_arguments(self, query: str) -> List[Dict]:
        """Search arguments by keyword in transcript"""
        results = []
        index = self._load_index()

        for item in index:
            arg = self.get_argument(item["id"])
            if arg and query.lower() in arg.get("transcript", "").lower():
                results.append(item)

        return results

    def get_stats(self) -> Dict:
        """Get database statistics"""
        index = self._load_index()

        total = len(index)
        if total == 0:
            return {"total_arguments": 0}

        # Count winners
        winner_counts = {}
        for item in index:
            winner = item.get("winner", "Unknown")
            winner_counts[winner] = winner_counts.get(winner, 0) + 1

        return {
            "total_arguments": total,
            "winner_distribution": winner_counts,
            "latest_timestamp": index[0]["timestamp"] if index else None
        }
