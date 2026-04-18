"""
Simple audio synthesizer for jam accompaniment.
Generates bass, chord stabs, and basic drum patterns.
"""

import numpy as np
from typing import Optional

from .theory import NOTE_NAMES, note_to_index, parse_chord_name, CHORD_TYPES


# Note frequencies (A4 = 440Hz)
def note_freq(note_name: str, octave: int = 4) -> float:
    """Get frequency of a note."""
    idx = note_to_index(note_name)
    midi = (octave + 1) * 12 + idx
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def generate_sine(freq: float, duration: float, sr: int,
                  amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def generate_square(freq: float, duration: float, sr: int,
                     amplitude: float = 0.3) -> np.ndarray:
    """Generate a square wave (rich harmonic content for bass)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sign(np.sin(2 * np.pi * freq * t))


def generate_noise(duration: float, sr: int,
                    amplitude: float = 0.3) -> np.ndarray:
    """Generate white noise (for hi-hats/snare)."""
    n = int(sr * duration)
    return amplitude * np.random.uniform(-1, 1, n)


def apply_envelope(signal: np.ndarray, attack: float = 0.01,
                    decay: float = 0.1, sustain: float = 0.7,
                    release: float = 0.1, sr: int = 22050) -> np.ndarray:
    """Apply ADSR envelope to a signal."""
    n = len(signal)
    env = np.ones(n)

    attack_samples = min(int(attack * sr), n)
    decay_samples = min(int(decay * sr), n - attack_samples)
    release_samples = min(int(release * sr), n)

    # Attack
    env[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Decay
    decay_end = attack_samples + decay_samples
    env[attack_samples:decay_end] = np.linspace(1, sustain, decay_samples)
    # Sustain (already set to sustain level)
    env[decay_end:-release_samples] = sustain
    # Release
    if release_samples > 0:
        env[-release_samples:] = np.linspace(sustain, 0, release_samples)

    return signal * env


def synth_bass_note(chord_name: str, duration: float, sr: int = 22050) -> np.ndarray:
    """Synthesize a bass note from a chord name (root note, octave 2)."""
    root, _ = parse_chord_name(chord_name)
    freq = note_freq(root, octave=2)
    signal = generate_square(freq, duration, sr, amplitude=0.4)
    # Add fundamental sine for warmth
    signal += generate_sine(freq, duration, sr, amplitude=0.2)
    return apply_envelope(signal, attack=0.005, decay=0.05,
                          sustain=0.6, release=0.1, sr=sr)


def synth_chord_stab(chord_name: str, duration: float,
                      sr: int = 22050) -> np.ndarray:
    """Synthesize a chord stab from chord name."""
    root, quality = parse_chord_name(chord_name)
    root_idx = note_to_index(root)
    intervals = CHORD_TYPES.get(quality, CHORD_TYPES['maj'])

    signal = np.zeros(int(sr * duration))
    for interval in intervals:
        note_idx = (root_idx + interval) % 12
        # Use octave 4 for chord tones
        octave = 4 if interval <= 7 else 5
        freq = note_freq(NOTE_NAMES[note_idx], octave=octave)
        signal += generate_sine(freq, duration, sr, amplitude=0.15)

    return apply_envelope(signal, attack=0.005, decay=0.1,
                          sustain=0.4, release=0.05, sr=sr)


def synth_kick(sr: int = 22050) -> np.ndarray:
    """Synthesize a kick drum."""
    duration = 0.15
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Pitch sweep from 150Hz down to 50Hz
    freq_start, freq_end = 150, 50
    phase = 2 * np.pi * (freq_start * t + (freq_end - freq_start) * t**2 / (2 * duration))
    signal = np.sin(phase)
    return apply_envelope(signal, attack=0.001, decay=0.1,
                          sustain=0.0, release=0.05, sr=sr)


def synth_hihat(duration: float = 0.05, sr: int = 22050) -> np.ndarray:
    """Synthesize a hi-hat (filtered noise)."""
    noise = generate_noise(duration, sr, amplitude=0.2)
    # Simple high-pass by subtracting a smoothed version
    kernel_size = max(3, int(sr * 0.003))
    smoothed = np.convolve(noise, np.ones(kernel_size) / kernel_size, mode='same')
    return apply_envelope(noise - smoothed, attack=0.001, decay=0.03,
                          sustain=0.0, release=0.02, sr=sr)


def synth_snare(sr: int = 22050) -> np.ndarray:
    """Synthesize a snare drum (tone + noise)."""
    duration = 0.12
    # Tone component
    tone = generate_sine(200, duration, sr, amplitude=0.3)
    # Noise component
    noise = generate_noise(duration, sr, amplitude=0.4)
    signal = tone + noise
    return apply_envelope(signal, attack=0.001, decay=0.08,
                          sustain=0.1, release=0.04, sr=sr)


class AccompanimentGenerator:
    """Generates full accompaniment from a chord progression."""

    def __init__(self, sr: int = 22050, tempo: int = 120):
        self.sr = sr
        self.tempo = tempo
        self.beat_duration = 60.0 / tempo

    def generate_bar(self, chord: str, style: str = "jazz") -> np.ndarray:
        """
        Generate one bar of accompaniment for a given chord.
        Returns audio array for one measure (4 beats).
        """
        beat_dur = self.beat_duration
        bar_dur = beat_dur * 4
        bar_samples = int(self.sr * bar_dur)
        bar = np.zeros(bar_samples)

        if style == "jazz":
            # Walking bass on beats 1-3, chord stab on 2 & 4
            for beat in range(3):
                start = int(beat * beat_dur * self.sr)
                bass = synth_bass_note(chord, beat_dur * 0.8, self.sr)
                end = min(start + len(bass), bar_samples)
                bar[start:end] += bass[:end - start]

            # Chord stabs on beats 2 and 4
            for beat_idx in [1, 3]:
                start = int(beat_idx * beat_dur * self.sr)
                stab = synth_chord_stab(chord, beat_dur * 0.5, self.sr)
                end = min(start + len(stab), bar_samples)
                bar[start:end] += stab[:end - start]

            # Hi-hat on every beat
            for beat in range(4):
                start = int(beat * beat_dur * self.sr)
                hh = synth_hihat(sr=self.sr)
                end = min(start + len(hh), bar_samples)
                bar[start:end] += hh[:end - start]

        elif style == "blues":
            # Bass on 1 and 3, chords on 2 and 4
            for beat in [0, 2]:
                start = int(beat * beat_dur * self.sr)
                bass = synth_bass_note(chord, beat_dur, self.sr)
                end = min(start + len(bass), bar_samples)
                bar[start:end] += bass[:end - start]

            for beat in [1, 3]:
                start = int(beat * beat_dur * self.sr)
                stab = synth_chord_stab(chord, beat_dur * 0.6, self.sr)
                end = min(start + len(stab), bar_samples)
                bar[start:end] += stab[:end - start]

            # Kick on 1, snare on 3
            kick_start = 0
            kick = synth_kick(self.sr)
            end = min(kick_start + len(kick), bar_samples)
            bar[kick_start:end] += kick[:end - kick_start]

            snare_start = int(2 * beat_dur * self.sr)
            snare = synth_snare(self.sr)
            end = min(snare_start + len(snare), bar_samples)
            bar[snare_start:end] += snare[:end - snare_start]

        else:  # rock / default
            # Power chords with driving rhythm
            for beat in range(4):
                start = int(beat * beat_dur * self.sr)
                # Downbeat: kick + bass
                if beat % 2 == 0:
                    kick = synth_kick(self.sr)
                    end = min(start + len(kick), bar_samples)
                    bar[start:end] += kick[:end - kick_start]

                    bass = synth_bass_note(chord, beat_dur * 0.9, self.sr)
                    end = min(start + len(bass), bar_samples)
                    bar[start:end] += bass[:end - start]
                else:
                    snare = synth_snare(self.sr)
                    end = min(start + len(snare), bar_samples)
                    bar[start:end] += snare[:end - start]

                stab = synth_chord_stab(chord, beat_dur * 0.8, self.sr)
                end = min(start + len(stab), bar_samples)
                bar[start:end] += stab[:end - start]

        # Soft clip
        bar = np.clip(bar, -0.95, 0.95)
        return bar

    def generate_progression(self, chords: list[str],
                              style: str = "jazz") -> np.ndarray:
        """Generate accompaniment for a full chord progression."""
        parts = []
        for chord in chords:
            parts.append(self.generate_bar(chord, style))
        return np.concatenate(parts) if parts else np.array([])
