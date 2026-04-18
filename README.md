# JamMate 🎸

**AI Jamming Partner** — You play, it listens, it jams with you.

JamMate listens to your instrument in real-time, recognizes chords you're playing,
predicts where the progression is going, and generates accompanying parts to jam along.

## How It Works

```
┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Audio   │───▶│    Chord     │───▶│   Progression  │───▶│    Jam       │
│  Capture │    │  Recognition │    │   Prediction   │    │  Generation  │
│  (mic)   │    │  (librosa)   │    │   (MiMo AI)    │    │  (synth)     │
└──────────┘    └──────────────┘    └───────────────┘    └──────────────┘
```

## Architecture

### Core Modules

| Module | Purpose |
|--------|---------|
| `chord_recognition.py` | Real-time chord detection from audio input using chroma features |
| `mimo_client.py` | MiMo AI client for chord progression prediction |
| `jam_engine.py` | Orchestrates the jam loop: listen → recognize → predict → generate |
| `audio_capture.py` | Microphone input with configurable buffer/sampling |
| `synth.py` | Audio synthesis for accompaniment (bass, chords, drums) |
| `theory.py` | Music theory: chord types, scales, progressions |
| `ui.py` | Terminal UI showing current chord, predicted next, and jam status |

### Jam Loop

1. **Capture** audio chunk (default 2s window)
2. **Recognize** chord from chroma features
3. **Build context** from last N recognized chords
4. **Predict** next chords using MiMo
5. **Generate** accompaniment audio
6. **Play** generated audio through speakers
7. **Repeat**

## Setup

```bash
pip install -r requirements.txt
```

### Configuration

Set your MiMo API endpoint in `config.yaml`:

```yaml
mimo:
  api_url: "https://your-mimo-endpoint"
  api_key: "your-key"
  model: "mimo-audio"

audio:
  sample_rate: 22050
  chunk_duration: 2.0
  channels: 1

jam:
  lookback_chords: 8    # how many past chords for context
  predict_count: 4      # how many chords to predict ahead
  tempo: 120
  style: "jazz"         # jazz, blues, rock, funk
```

## Usage

```bash
python -m jammate
```

## Development Roadmap

- [x] Project structure & design
- [ ] Chord recognition from audio files
- [ ] Real-time audio capture
- [ ] MiMo integration for progression prediction
- [ ] Basic accompaniment synthesis
- [ ] Real-time jam loop
- [ ] Terminal UI
- [ ] Multiple jam styles
- [ ] Record & export sessions
