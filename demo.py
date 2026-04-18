#!/usr/bin/env python3
"""
Quick demo — generate and play a jam progression without a microphone.
Usage: python3 demo.py [style]
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jammate.synth import AccompanimentGenerator
from jammate.theory import COMMON_PROGRESSIONS, DIATONIC_MAJOR
from jammate.visualizer import render_progression, print_chord_wheel


def main():
    style = sys.argv[1] if len(sys.argv) > 1 else 'jazz'
    tempo = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    key = sys.argv[3] if len(sys.argv) > 3 else 'C'

    print(f"\n🎵 JamMate Demo — {style} in {key} @ {tempo} BPM\n")

    # Get diatonic chords for the key
    diatonic = DIATONIC_MAJOR.get(key, DIATONIC_MAJOR['C'])

    # Map progression degrees to actual chords
    degrees = COMMON_PROGRESSIONS.get(style, COMMON_PROGRESSIONS['jazz'])
    chords = [diatonic[d - 1] for d in degrees]

    # Show chord wheel
    print_chord_wheel(key)

    # Show progression
    print(f"  Progression ({style}):")
    print(render_progression(chords, key))
    print()

    # Generate audio
    synth = AccompanimentGenerator(tempo=tempo)
    audio = synth.generate_progression(chords, style)

    duration = len(audio) / synth.sr
    print(f"  Generated {duration:.1f}s of audio ({len(chords)} bars)")

    # Try to play, fallback to saving file
    try:
        from jammate.audio_capture import play_audio
        print("  🔊 Playing...")
        play_audio(audio, synth.sr)
        print("  ✓ Done!")
    except Exception as e:
        # Save to WAV
        import wave
        out = f"demo_{style}_{key}.wav"
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(out, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(synth.sr)
            wf.writeframes(pcm.tobytes())
        print(f"  [Audio] Playback unavailable ({e})")
        print(f"  Saved to: {out}")

    # Generate prediction display (simulate MiMo)
    print(f"\n  🔮 Would predict next (with MiMo):")
    next_chords = chords[-4:] if len(chords) >= 4 else chords
    print(render_progression(next_chords, key))
    print()


if __name__ == '__main__':
    main()
