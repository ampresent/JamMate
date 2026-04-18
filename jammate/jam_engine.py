"""
Jam Engine — the core loop that ties everything together.

Captures audio → recognizes chords → predicts next → generates jam → plays
"""

import time
import signal
import sys
import yaml
import numpy as np
from typing import Optional
from pathlib import Path

from .chord_recognition import detect_chords_from_audio
from .mimo_client import MiMoClient, create_client_from_config
from .progression_gen import ProgressionGenerator
from .synth import AccompanimentGenerator
from .audio_capture import AudioCapture, play_audio
from .theory import get_key_from_chords
from .visualizer import render_jam_status


class JamSession:
    """A single jam session with history tracking."""

    def __init__(self):
        self.chord_history: list[dict] = []  # {chord, time, confidence}
        self.start_time = time.time()
        self.bpm = 120
        self.style = "jazz"

    @property
    def chords_only(self) -> list[str]:
        return [c['chord'] for c in self.chord_history]

    @property
    def detected_key(self) -> Optional[str]:
        return get_key_from_chords(self.chords_only)

    def add_chord(self, chord: str, confidence: float):
        self.chord_history.append({
            'chord': chord,
            'time': time.time() - self.start_time,
            'confidence': confidence,
        })


class JamEngine:
    """Main engine that runs the jam loop."""

    def __init__(self, config: dict):
        self.config = config

        # Audio settings
        audio_cfg = config.get('audio', {})
        self.sample_rate = audio_cfg.get('sample_rate', 22050)
        self.chunk_duration = audio_cfg.get('chunk_duration', 2.0)

        # Jam settings
        jam_cfg = config.get('jam', {})
        self.lookback = jam_cfg.get('lookback_chords', 8)
        self.predict_count = jam_cfg.get('predict_count', 4)
        self.tempo = jam_cfg.get('tempo', 120)
        self.style = jam_cfg.get('style', 'jazz')
        self.min_confidence = jam_cfg.get('min_confidence', 0.3)

        # Initialize components
        self.capture = AudioCapture(
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
        )
        self.synth = AccompanimentGenerator(
            sr=self.sample_rate,
            tempo=self.tempo,
        )
        self.mimo: Optional[MiMoClient] = None
        self.local_gen = ProgressionGenerator(key='C', style=self.style)
        self.session = JamSession()
        self.session.bpm = self.tempo
        self.session.style = self.style
        self._running = False

    def connect_mimo(self):
        """Connect to MiMo AI for chord prediction."""
        self.mimo = create_client_from_config(self.config)
        print(f"[Jam] MiMo connected: {self.config.get('mimo', {}).get('api_url', 'default')}")

    def run(self):
        """Main jam loop."""
        self._running = True

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\n[Jam] Stopping...")
            self._running = False
        signal.signal(signal.SIGINT, signal_handler)

        print("=" * 50)
        print(f"  JamMate — {self.style} jam @ {self.tempo} BPM")
        print(f"  Press Ctrl+C to stop")
        print("=" * 50)

        if self.mimo:
            print(f"  MiMo: connected")
        else:
            print(f"  MiMo: offline (using local fallback)")

        print()

        beat_count = 0

        while self._running:
            # 1. Capture audio
            print(f"\r🎸 Listening...", end="", flush=True)
            audio_chunk = self.capture.capture_chunk()

            if audio_chunk is None:
                continue

            # 2. Recognize chord
            detections = detect_chords_from_audio(
                audio_chunk, self.sample_rate,
                window_sec=min(self.chunk_duration, 2.0),
                hop_sec=min(self.chunk_duration * 0.5, 1.0),
                min_confidence=self.min_confidence,
            )

            if not detections:
                continue

            # Take the most confident detection
            best = max(detections, key=lambda d: d['confidence'])
            chord = best['chord']
            confidence = best['confidence']

            # 3. Add to history
            self.session.add_chord(chord, confidence)
            beat_count += 1

            # 4. Display with visualizer
            history = self.session.chords_only[-self.lookback:]
            key = self.session.detected_key

            # Build display
            display = render_jam_status(chord, confidence, history,
                                         [], key, beat=beat_count)
            print(f"\033[2J\033[H{display}", end="", flush=True)

            # 5. Predict next chords (every 2 detections to reduce API calls)
            if beat_count % 2 == 0:
                predicted = self._predict_chords(history, key)
                if predicted:
                    pred_str = ' → '.join(predicted)
                    print(f"\n   🔮 Next: {pred_str}", end="", flush=True)

                    # 6. Generate and play accompaniment
                    try:
                        jam_audio = self.synth.generate_progression(
                            predicted[:2],  # Play 2 bars ahead
                            style=self.style,
                        )
                        if len(jam_audio) > 0:
                            play_audio(jam_audio, self.sample_rate)
                    except Exception as e:
                        print(f"\n   [Synth] Error: {e}")

        print(f"\n\n[Jam] Session ended. {len(self.session.chord_history)} chords played.")

    def _predict_chords(self, history: list[str],
                         key: Optional[str]) -> list[str]:
        """Predict next chords using MiMo or fallback."""
        if self.mimo:
            try:
                return self.mimo.predict_next_chords(
                    history,
                    style=self.style,
                    count=self.predict_count,
                    key=key,
                )
            except Exception as e:
                print(f"\n   [MiMo] Error: {e}, using fallback")

        # Fallback to local generator
        self.local_gen.key = key or 'C'
        return self.local_gen.generate_next(self.predict_count, history)

    def analyze_file(self, filepath: str) -> dict:
        """
        Analyze an audio file (offline mode).
        Returns chord progression and predictions.
        """
        from .chord_recognition import recognize_chord_from_file

        print(f"[Jam] Analyzing: {filepath}")
        chords = recognize_chord_from_file(filepath, self.min_confidence)

        chord_names = [c['chord'] for c in chords]
        key = get_key_from_chords(chord_names)

        result = {
            'file': filepath,
            'detected_chords': chords,
            'chord_names': chord_names,
            'estimated_key': key,
            'predictions': [],
        }

        if chord_names and self.mimo:
            try:
                predicted = self.mimo.predict_next_chords(
                    chord_names, self.style, self.predict_count, key,
                )
                result['predictions'] = predicted
            except Exception as e:
                print(f"[MiMo] Prediction error: {e}")

        return result


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def create_default_config(path: str = "config.yaml"):
    """Create a default config file."""
    default = {
        'mimo': {
            'api_url': 'https://api.openai-compatible.com/v1',
            'api_key': '',
            'model': 'mimo-v2-pro',
        },
        'audio': {
            'sample_rate': 22050,
            'chunk_duration': 2.0,
            'channels': 1,
        },
        'jam': {
            'lookback_chords': 8,
            'predict_count': 4,
            'tempo': 120,
            'style': 'jazz',
            'min_confidence': 0.3,
        },
    }
    with open(path, 'w') as f:
        yaml.dump(default, f, default_flow_style=False, allow_unicode=True)
    print(f"[Config] Created default config at {path}")
