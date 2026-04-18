"""
Session recorder — save jam sessions as WAV + metadata JSON.
"""

import json
import wave
import time
import numpy as np
from pathlib import Path
from typing import Optional


class SessionRecorder:
    """Records jam sessions to disk."""

    def __init__(self, output_dir: str = "recordings", sr: int = 22050):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sr = sr
        self.chunks: list[np.ndarray] = []
        self.metadata: list[dict] = []
        self.start_time: Optional[float] = None

    def start(self):
        """Start recording."""
        self.chunks = []
        self.metadata = []
        self.start_time = time.time()
        print(f"[Recorder] Started recording")

    def add_audio(self, samples: np.ndarray, label: str = ""):
        """Add an audio chunk with metadata."""
        self.chunks.append(samples.copy())
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.metadata.append({
            'time': round(elapsed, 2),
            'label': label,
            'samples': len(samples),
        })

    def add_chord_event(self, chord: str, confidence: float):
        """Log a chord detection event."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.metadata.append({
            'time': round(elapsed, 2),
            'event': 'chord_detected',
            'chord': chord,
            'confidence': round(confidence, 3),
        })

    def add_prediction_event(self, predicted: list[str]):
        """Log a prediction event."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.metadata.append({
            'time': round(elapsed, 2),
            'event': 'prediction',
            'chords': predicted,
        })

    def stop(self, filename: Optional[str] = None) -> str:
        """
        Stop recording and save to files.
        Returns the path to the saved WAV file.
        """
        if not self.chunks:
            print("[Recorder] No audio to save")
            return ""

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if filename is None:
            filename = f"jam_{timestamp}"

        wav_path = self.output_dir / f"{filename}.wav"
        json_path = self.output_dir / f"{filename}.json"

        # Concatenate and save WAV
        audio = np.concatenate(self.chunks)
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767).astype(np.int16)

        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(pcm.tobytes())

        # Save metadata
        meta = {
            'timestamp': timestamp,
            'duration': round(time.time() - self.start_time, 2) if self.start_time else 0,
            'sample_rate': self.sr,
            'events': self.metadata,
            'total_chords_detected': sum(
                1 for m in self.metadata if m.get('event') == 'chord_detected'
            ),
            'total_predictions': sum(
                1 for m in self.metadata if m.get('event') == 'prediction'
            ),
        }

        with open(json_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        duration = meta['duration']
        print(f"[Recorder] Saved: {wav_path} ({duration:.1f}s)")
        print(f"[Recorder] Metadata: {json_path}")

        return str(wav_path)

    @property
    def is_recording(self) -> bool:
        return self.start_time is not None and self.chunks
