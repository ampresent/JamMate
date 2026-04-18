"""
Real-time audio capture from microphone.
Captures audio chunks for chord recognition.
"""

import subprocess
import struct
import os
import tempfile
import numpy as np
from typing import Optional, Callable


class AudioCapture:
    """
    Captures audio from the default microphone.

    Supports two backends:
    - arecord (ALSA, Linux)
    - pyaudio (if installed)
    """

    def __init__(self, sample_rate: int = 22050, channels: int = 1,
                 chunk_duration: float = 2.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self._recording = False
        self._tmp_dir = tempfile.mkdtemp(prefix="jammate_")

    def capture_chunk(self) -> Optional[np.ndarray]:
        """
        Capture one chunk of audio from the microphone.
        Returns numpy array of samples, or None if capture failed.
        """
        tmp_file = os.path.join(self._tmp_dir, "chunk.wav")

        try:
            # Use arecord (ALSA) on Linux
            cmd = [
                'arecord',
                '-f', 'S16_LE',       # 16-bit little-endian
                '-r', str(self.sample_rate),
                '-c', str(self.channels),
                '-t', 'wav',
                '-d', str(int(self.chunk_duration)),
                '-q',                  # quiet mode
                tmp_file,
            ]
            subprocess.run(cmd, timeout=int(self.chunk_duration) + 2,
                          check=True)

            return self._read_wav(tmp_file)

        except FileNotFoundError:
            print("[Audio] arecord not found. Trying ffmpeg...")
            return self._capture_with_ffmpeg(tmp_file)
        except subprocess.TimeoutExpired:
            print("[Audio] Capture timeout")
            return None
        except Exception as e:
            print(f"[Audio] Capture error: {e}")
            return None
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    def _capture_with_ffmpeg(self, tmp_file: str) -> Optional[np.ndarray]:
        """Fallback: capture with ffmpeg."""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-f', 'alsa',          # ALSA input
                '-i', 'default',
                '-t', str(int(self.chunk_duration)),
                '-ar', str(self.sample_rate),
                '-ac', str(self.channels),
                '-sample_fmt', 's16',
                '-loglevel', 'error',
                tmp_file,
            ]
            subprocess.run(cmd, timeout=int(self.chunk_duration) + 5,
                          check=True)
            return self._read_wav(tmp_file)
        except Exception as e:
            print(f"[Audio] ffmpeg capture error: {e}")
            return None

    def _read_wav(self, filepath: str) -> np.ndarray:
        """Read WAV file into numpy array."""
        import wave
        with wave.open(filepath, 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            samples = np.array(
                struct.unpack(f'<{n_frames * n_channels}h', raw),
                dtype=np.float32
            )
            samples /= 32768.0
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            return samples

    def record_to_file(self, output_path: str, duration: float = 10.0):
        """Record audio directly to a file."""
        try:
            cmd = [
                'arecord',
                '-f', 'S16_LE',
                '-r', str(self.sample_rate),
                '-c', str(self.channels),
                '-t', 'wav',
                '-d', str(int(duration)),
                output_path,
            ]
            subprocess.run(cmd, timeout=int(duration) + 2, check=True)
            print(f"[Audio] Saved to {output_path}")
        except Exception as e:
            print(f"[Audio] Record error: {e}")

    def cleanup(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)


def play_audio(samples: np.ndarray, sr: int = 22050):
    """Play audio samples through speakers using aplay."""
    import tempfile

    # Convert to 16-bit PCM
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    try:
        # Write WAV
        import wave
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

        subprocess.run(['aplay', '-q', tmp.name],
                       timeout=10, check=False)
    except Exception as e:
        print(f"[Audio] Playback error: {e}")
    finally:
        os.unlink(tmp.name)
