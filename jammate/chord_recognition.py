"""
Chord recognition from audio using chroma feature analysis.

Takes raw audio (numpy array or file path) and returns detected chord(s)
with confidence scores.
"""

import numpy as np
from typing import Optional

from .theory import NOTE_NAMES, CHORD_TYPES, build_chord_vector, chord_similarity


def compute_chroma(y: np.ndarray, sr: int, hop_length: int = 512,
                   n_fft: int = 4096) -> np.ndarray:
    """
    Compute chromagram from audio signal.
    Returns (12, T) array of chroma features over time.
    Uses STFT-based chroma extraction.
    """
    # STFT
    window = np.hanning(n_fft)
    num_frames = 1 + (len(y) - n_fft) // hop_length
    chroma = np.zeros((12, num_frames))

    for i in range(num_frames):
        start = i * hop_length
        frame = y[start:start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))

        # Map frequency bins to pitch classes
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        for j, freq in enumerate(freqs):
            if freq < 80 or freq > 5000:  # Skip extreme frequencies
                continue
            # Convert frequency to MIDI note, then to pitch class
            if freq > 0:
                midi = 69 + 12 * np.log2(freq / 440.0)
                pitch_class = int(round(midi)) % 12
                chroma[pitch_class, i] += spectrum[j] ** 2

    # Normalize each frame
    for i in range(chroma.shape[1]):
        norm = np.sum(chroma[:, i])
        if norm > 0:
            chroma[:, i] /= norm

    return chroma


def detect_chord_from_chroma(chroma_frame: np.ndarray,
                              min_confidence: float = 0.3) -> Optional[tuple[str, float]]:
    """
    Detect chord from a single chroma frame.
    Returns (chord_name, confidence) or None if below threshold.
    Tests major and minor templates against the chroma vector.
    """
    best_chord = None
    best_score = 0.0

    for root_idx in range(12):
        for chord_type in ['maj', 'min']:
            template = build_chord_vector(root_idx, chord_type)
            score = chord_similarity(chroma_frame.tolist(), template)
            if score > best_score:
                best_score = score
                note_name = NOTE_NAMES[root_idx]
                if chord_type == 'min':
                    best_chord = f"{note_name}m"
                else:
                    best_chord = note_name

    if best_score >= min_confidence:
        return best_chord, best_score
    return None


def detect_chords_from_audio(y: np.ndarray, sr: int,
                              window_sec: float = 2.0,
                              hop_sec: float = 0.5,
                              min_confidence: float = 0.3) -> list[dict]:
    """
    Detect chords from audio signal with timestamps.

    Args:
        y: Audio signal (mono)
        sr: Sample rate
        window_sec: Analysis window duration
        hop_sec: Hop between windows
        min_confidence: Minimum similarity threshold

    Returns:
        List of {'start': float, 'end': float, 'chord': str, 'confidence': float}
    """
    hop_length = int(sr * hop_sec)
    n_fft = int(sr * window_sec)
    # Round n_fft to nearest power of 2 for efficiency
    n_fft = 1 << (n_fft - 1).bit_length()

    chroma = compute_chroma(y, sr, hop_length=hop_length, n_fft=n_fft)

    results = []
    for i in range(chroma.shape[1]):
        frame = chroma[:, i]
        detection = detect_chord_from_chroma(frame, min_confidence)
        if detection:
            chord_name, confidence = detection
            start_time = i * hop_sec
            results.append({
                'start': round(start_time, 2),
                'end': round(start_time + window_sec, 2),
                'chord': chord_name,
                'confidence': round(confidence, 3),
            })

    return _merge_adjacent_chords(results)


def _merge_adjacent_chords(chords: list[dict],
                            min_duration: float = 0.3) -> list[dict]:
    """Merge consecutive identical chords and filter very short detections."""
    if not chords:
        return []

    merged = [chords[0].copy()]
    for c in chords[1:]:
        if c['chord'] == merged[-1]['chord']:
            merged[-1]['end'] = c['end']
            merged[-1]['confidence'] = max(merged[-1]['confidence'], c['confidence'])
        else:
            merged.append(c.copy())

    # Filter out too-short detections
    return [c for c in merged if c['end'] - c['start'] >= min_duration]


def recognize_chord_from_file(filepath: str,
                               min_confidence: float = 0.3) -> list[dict]:
    """
    Recognize chords from an audio file.
    Supports WAV, FLAC, MP3 (requires soundfile or pydub).
    """
    import wave
    import struct

    if filepath.endswith('.wav'):
        with wave.open(filepath, 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

            # Convert to float numpy array
            if wf.getsampwidth() == 2:
                fmt = f'<{n_frames * n_channels}h'
                samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
                samples /= 32768.0
            else:
                raise ValueError("Only 16-bit WAV supported currently")

            # Mix to mono if stereo
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            return detect_chords_from_audio(samples, sr,
                                             min_confidence=min_confidence)
    else:
        raise ValueError(f"Unsupported format: {filepath}. Use WAV for now.")
