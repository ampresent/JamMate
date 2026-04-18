"""
Local chord progression generator — music theory based fallback when MiMo is offline.
Uses voice leading rules and common progression patterns.
"""

import random
from typing import Optional
from .theory import (
    DIATONIC_MAJOR, COMMON_PROGRESSIONS,
    parse_chord_name, note_to_index, NOTE_NAMES,
)


# Weighted transition probabilities (scale degree -> [(next_degree, weight)])
# Higher weight = more likely transition
TRANSITION_WEIGHTS = {
    # degree 1-indexed
    1:  [(4, 3), (5, 4), (6, 3), (2, 2)],
    2:  [(5, 5), (1, 1)],
    3:  [(6, 3), (4, 2), (1, 1)],
    4:  [(5, 4), (1, 3), (2, 2)],
    5:  [(1, 5), (6, 3), (4, 2)],
    6:  [(2, 3), (4, 3), (5, 2), (1, 1)],
    7:  [(1, 5), (5, 1)],
}


def weighted_choice(choices: list[tuple[int, float]]) -> int:
    """Pick from [(value, weight)] pairs."""
    total = sum(w for _, w in choices)
    r = random.uniform(0, total)
    cumulative = 0
    for value, weight in choices:
        cumulative += weight
        if r <= cumulative:
            return value
    return choices[-1][0]


class ProgressionGenerator:
    """Generates chord progressions using local music theory."""

    def __init__(self, key: str = 'C', style: str = 'jazz'):
        self.key = key
        self.style = style
        self.diatonic = DIATONIC_MAJOR.get(key, DIATONIC_MAJOR['C'])
        self._position = 1  # Current scale degree (1-indexed)

    def generate_next(self, count: int = 4,
                       history: Optional[list[str]] = None) -> list[str]:
        """
        Generate the next `count` chords.

        Strategy:
        1. If we have history, try to continue the pattern
        2. Use transition weights for voice-leading
        3. Apply style-specific rules
        """
        result = []

        # If history provided, find current position from last chord
        if history:
            last = history[-1]
            root, _ = parse_chord_name(last)
            diatonic_roots = [parse_chord_name(d)[0] for d in self.diatonic]
            if root in diatonic_roots:
                self._position = diatonic_roots.index(root) + 1

        for _ in range(count):
            transitions = TRANSITION_WEIGHTS.get(self._position, [(1, 1)])
            self._position = weighted_choice(transitions)

            # Apply style modifications
            chord = self._degree_to_chord(self._position)
            chord = self._apply_style(chord, self._position)
            result.append(chord)

        return result

    def generate_fixed(self, style: Optional[str] = None) -> list[str]:
        """Generate a standard progression for the style."""
        s = style or self.style
        degrees = COMMON_PROGRESSIONS.get(s, COMMON_PROGRESSIONS['jazz'])
        return [self._degree_to_chord(d) for d in degrees]

    def _degree_to_chord(self, degree: int) -> str:
        """Convert scale degree to chord name."""
        idx = (degree - 1) % 7
        return self.diatonic[idx]

    def _apply_style(self, chord: str, degree: int) -> str:
        """Apply style-specific chord modifications."""
        root, quality = parse_chord_name(chord)

        if self.style == 'jazz':
            # Add 7ths to most chords
            if quality == 'maj' and degree in [1, 4]:
                return f"{root}maj7"
            elif quality == 'min':
                return f"{root}m7"
            elif quality == 'maj' and degree == 5:
                return f"{root}7"

        elif self.style == 'blues':
            # All dominant 7ths
            if quality == 'maj':
                return f"{root}7"

        elif self.style == 'funk':
            # 9th chords
            if quality == 'maj':
                return f"{root}9"
            elif quality == 'min':
                return f"{root}m9"

        return chord

    def suggest_substitution(self, chord: str) -> list[str]:
        """
        Suggest chord substitutions for a given chord.
        Common jazz substitutions.
        """
        root, quality = parse_chord_name(chord)
        subs = []

        if quality in ('7', 'maj'):
            # Tritone substitution (dominant chords)
            root_idx = note_to_index(root)
            tritone_idx = (root_idx + 6) % 12
            tritone_note = NOTE_NAMES[tritone_idx]
            subs.append(f"{tritone_note}7")

        # Relative minor/major
        if quality == 'maj':
            # Major -> relative minor (down 3 semitones)
            root_idx = note_to_index(root)
            rel_idx = (root_idx - 3) % 12
            subs.append(f"{NOTE_NAMES[rel_idx]}m7")
        elif quality == 'min':
            root_idx = note_to_index(root)
            rel_idx = (root_idx + 3) % 12
            subs.append(f"{NOTE_NAMES[rel_idx]}maj7")

        return subs
