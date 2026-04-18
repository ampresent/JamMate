"""
JamMate — AI Jamming Partner

Usage:
    python -m jammate                    # Live jam mode
    python -m jammate --file audio.wav   # Analyze an audio file
    python -m jammate --demo             # Demo mode (no mic needed)
    python -m jammate --config           # Create default config
"""

import sys
import os

# Add parent dir to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jammate.jam_engine import JamEngine, load_config, create_default_config
from jammate.ui import (
    print_header, print_help, print_status,
    print_history, print_predictions, print_session_summary,
)
from jammate.recorder import SessionRecorder


def main():
    args = sys.argv[1:]

    if '--help' in args or '-h' in args:
        print_header()
        print_help()
        return

    if '--config' in args:
        config_path = 'config.yaml'
        for i, arg in enumerate(args):
            if arg == '--config' and i + 1 < len(args):
                config_path = args[i + 1]
        create_default_config(config_path)
        return

    config = load_config()

    # Override style from args
    for style in ['jazz', 'blues', 'rock', 'pop', 'funk']:
        if f'--{style}' in args:
            config.setdefault('jam', {})['style'] = style

    engine = JamEngine(config)

    # Try connecting to MiMo
    try:
        engine.connect_mimo()
    except Exception as e:
        print(f"[Jam] MiMo unavailable: {e}")
        print("[Jam] Running in offline mode (local fallback)")

    if '--file' in args:
        # File analysis mode
        idx = args.index('--file')
        if idx + 1 < len(args):
            filepath = args[idx + 1]
            result = engine.analyze_file(filepath)

            print_header()
            print(f"  File: {result['file']}")
            print(f"  Key:  {result['estimated_key'] or 'Unknown'}")
            print(f"\n  Detected chords:")
            for c in result['detected_chords']:
                print(f"    {c['start']:6.1f}s - {c['end']:6.1f}s  "
                      f"{c['chord']:8s} ({c['confidence']:.0%})")
            if result['predictions']:
                print(f"\n  🔮 Predicted next: {' → '.join(result['predictions'])}")
            print()
        else:
            print("Error: --file requires a path")
            sys.exit(1)

    elif '--demo' in args:
        # Demo mode — no microphone needed
        print_header()
        print("  🎵 Demo Mode (no microphone)\n")

        from jammate.synth import AccompanimentGenerator
        from jammate.audio_capture import play_audio

        style = config.get('jam', {}).get('style', 'jazz')
        tempo = config.get('jam', {}).get('tempo', 120)
        synth = AccompanimentGenerator(tempo=tempo)

        demo_chords = {
            'jazz':  ['Dm7', 'G7', 'Cmaj7', 'Am7', 'Dm7', 'G7', 'Cmaj7', 'Cmaj7'],
            'blues': ['C7', 'C7', 'C7', 'C7', 'F7', 'F7', 'C7', 'C7', 'G7', 'F7', 'C7', 'G7'],
            'rock':  ['C', 'C', 'G', 'G', 'Am', 'Am', 'F', 'F'],
            'pop':   ['C', 'G', 'Am', 'F', 'C', 'G', 'Am', 'F'],
            'funk':  ['E9', 'E9', 'A9', 'A9', 'E9', 'E9', 'A9', 'E9'],
        }

        chords = demo_chords.get(style, demo_chords['jazz'])
        print(f"  Style: {style} | Tempo: {tempo} BPM")
        print(f"  Progression: {' → '.join(chords)}\n")

        audio = synth.generate_progression(chords, style)
        print(f"  Generated {len(audio) / synth.sr:.1f}s of audio")
        print(f"  🔊 Playing...")

        try:
            play_audio(audio, synth.sr)
        except Exception as e:
            # Save to file as fallback
            import wave
            import numpy as np
            out_path = "demo_output.wav"
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            with wave.open(out_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(synth.sr)
                wf.writeframes(pcm.tobytes())
            print(f"  [Audio] Playback failed ({e}), saved to {out_path}")

    else:
        # Live jam mode
        print_header()
        engine.run()


if __name__ == '__main__':
    main()
