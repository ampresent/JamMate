"""
MiMo AI client for chord progression prediction.

Uses Xiaomi's MiMo multimodal model to analyze chord progressions
and predict what comes next in a jamming context.
"""

import json
import requests
from typing import Optional


class MiMoClient:
    """Client for MiMo API to predict chord progressions."""

    def __init__(self, api_url: str, api_key: str = "",
                 model: str = "mimo-v2-pro"):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model = model

    def predict_next_chords(self,
                            chord_history: list[str],
                            style: str = "jazz",
                            count: int = 4,
                            key: Optional[str] = None) -> list[str]:
        """
        Predict the next chords based on what's been played.

        Args:
            chord_history: List of recently played chord names
            style: Music style (jazz, blues, rock, funk, pop)
            count: Number of chords to predict
            key: Current key (auto-detected if None)

        Returns:
            List of predicted chord names
        """
        if not chord_history:
            return self._default_opening(style, count)

        history_str = ' → '.join(chord_history[-8:])  # last 8 chords max

        prompt = f"""You are an expert {style} musician and music theorist.

A musician is jamming and has played these chords in sequence:
{history_str}

{'The key is ' + key + '.' if key else 'Determine the most likely key from context.'}

Based on {style} harmony conventions and the chord progression so far,
predict the next {count} chord(s) that would naturally follow.

Rules:
- Return ONLY a JSON array of chord name strings, nothing else
- Use standard chord notation (e.g., "Am", "Cmaj7", "G7", "Dm7b5")
- Make it musically coherent and interesting for jamming
- Consider voice leading and common {style} substitutions

Example response format: ["Dm7", "G7", "Cmaj7", "Am7"]"""

        try:
            response = self._call_api(prompt)
            chords = self._parse_chord_response(response)
            return chords[:count] if chords else self._fallback(history_str, style, count)
        except Exception as e:
            print(f"[MiMo] Prediction error: {e}")
            return self._fallback(history_str, style, count)

    def describe_progression(self, chords: list[str],
                              style: str = "jazz") -> str:
        """Get a human-readable description of a chord progression."""
        chord_str = ' → '.join(chords)
        prompt = f"""In one short sentence, describe this {style} chord progression:
{chord_str}
Be concise and musical. Example: 'Classic ii-V-I turnaround' or 'Blues walkup to the IV'"""

        try:
            return self._call_api(prompt).strip()
        except Exception:
            return f"{style} progression: {chord_str}"

    def _call_api(self, prompt: str) -> str:
        """Make API call to MiMo endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a music theory expert. Respond only with valid JSON when asked for chords."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 256,
        }

        resp = requests.post(
            f"{self.api_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()

        data = resp.json()
        return data['choices'][0]['message']['content']

    def _parse_chord_response(self, response: str) -> list[str]:
        """Extract chord list from MiMo's response."""
        # Try to find JSON array in response
        response = response.strip()

        # Remove markdown code blocks if present
        if '```' in response:
            parts = response.split('```')
            for part in parts:
                part = part.strip()
                if part.startswith('json'):
                    part = part[4:].strip()
                if part.startswith('['):
                    response = part
                    break

        # Find the array brackets
        start = response.find('[')
        end = response.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            try:
                chords = json.loads(json_str)
                if isinstance(chords, list) and all(isinstance(c, str) for c in chords):
                    return chords
            except json.JSONDecodeError:
                pass

        # Fallback: look for comma-separated chord names
        import re
        # Match patterns like: "Am7", "C", "G7"
        matches = re.findall(r'"([A-G][#b]?[^"]*)"', response)
        if matches:
            return matches

        return []

    def _default_opening(self, style: str, count: int) -> list[str]:
        """Default opening chords when there's no history."""
        openings = {
            'jazz':  ['Dm7', 'G7', 'Cmaj7', 'Am7'],
            'blues': ['C7', 'C7', 'F7', 'C7'],
            'rock':  ['C', 'F', 'G', 'C'],
            'pop':   ['C', 'G', 'Am', 'F'],
            'funk':  ['E9', 'E9', 'A9', 'E9'],
        }
        return openings.get(style, openings['jazz'])[:count]

    def _fallback(self, history: str, style: str, count: int) -> list[str]:
        """Simple fallback when API fails - use local theory."""
        from .theory import COMMON_PROGRESSIONS

        prog = COMMON_PROGRESSIONS.get(style, COMMON_PROGRESSIONS['jazz'])
        # Just cycle through the progression
        result = []
        for i in range(count):
            degree = prog[i % len(prog)]
            # Use C major scale chords as default
            diatonic = ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim']
            result.append(diatonic[degree - 1])
        return result


def create_client_from_config(config: dict) -> MiMoClient:
    """Create MiMoClient from config dict."""
    mimo_cfg = config.get('mimo', {})
    return MiMoClient(
        api_url=mimo_cfg.get('api_url', 'https://api.openai-compatible.com/v1'),
        api_key=mimo_cfg.get('api_key', ''),
        model=mimo_cfg.get('model', 'mimo-v2-pro'),
    )
