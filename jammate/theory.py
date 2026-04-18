"""
Music theory utilities for chord recognition and progression analysis.
"""

# Chromatic scale note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Chord templates: semitone intervals from root
CHORD_TYPES = {
    'maj':    [0, 4, 7],
    'min':    [0, 3, 7],
    'dim':    [0, 3, 6],
    'aug':    [0, 4, 8],
    '7':      [0, 4, 7, 10],
    'maj7':   [0, 4, 7, 11],
    'min7':   [0, 3, 7, 10],
    'dim7':   [0, 3, 6, 9],
    'm7b5':   [0, 3, 6, 10],
    'sus2':   [0, 2, 7],
    'sus4':   [0, 5, 7],
    '6':      [0, 4, 7, 9],
    'min6':   [0, 3, 7, 9],
    '9':      [0, 4, 7, 10, 14],
    'min9':   [0, 3, 7, 10, 14],
    'maj9':   [0, 4, 7, 11, 14],
    '11':     [0, 4, 7, 10, 14, 17],
    '13':     [0, 4, 7, 10, 14, 17, 21],
    'add9':   [0, 4, 7, 14],
    '5':      [0, 7],          # power chord
}

# Common chord progressions (scale degrees, 1-indexed)
COMMON_PROGRESSIONS = {
    'pop':     [1, 5, 6, 4],      # I-V-vi-IV
    'jazz':    [2, 5, 1],          # ii-V-I
    'blues':   [1, 1, 4, 1, 5, 4, 1, 5],  # 12-bar blues
    'folk':    [1, 4, 5, 1],      # I-IV-V-I
    'rock':    [1, 4, 5, 4],      # I-IV-V-IV
    'cliche':  [1, 6, 4, 5],      # I-vi-IV-V
    'minor':   [1, 4, 5, 1],      # i-iv-v-i (natural minor)
    'doo_wop': [1, 6, 4, 5],      # I-vi-IV-V
}

# Diatonic chords by key (major scale)
DIATONIC_MAJOR = {
    'C': ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim'],
    'G': ['G', 'Am', 'Bm', 'C', 'D', 'Em', 'F#dim'],
    'D': ['D', 'Em', 'F#m', 'G', 'A', 'Bm', 'C#dim'],
    'A': ['A', 'Bm', 'C#m', 'D', 'E', 'F#m', 'G#dim'],
    'E': ['E', 'F#m', 'G#m', 'A', 'B', 'C#m', 'D#dim'],
    'F': ['F', 'Gm', 'Am', 'Bb', 'C', 'Dm', 'Edim'],
    'Bb': ['Bb', 'Cm', 'Dm', 'Eb', 'F', 'Gm', 'Adim'],
}


def note_to_index(note: str) -> int:
    """Convert note name to chromatic index (0-11)."""
    note = note.strip()
    if note in NOTE_NAMES:
        return NOTE_NAMES.index(note)
    if note in NOTE_NAMES_FLAT:
        return NOTE_NAMES_FLAT.index(note)
    raise ValueError(f"Unknown note name: {note}")


def index_to_note(idx: int) -> str:
    """Convert chromatic index to note name (sharp notation)."""
    return NOTE_NAMES[idx % 12]


def build_chord_vector(root_idx: int, chord_type: str) -> list[float]:
    """Build a 12-dim binary vector for a chord."""
    intervals = CHORD_TYPES.get(chord_type, CHORD_TYPES['maj'])
    vec = [0.0] * 12
    for interval in intervals:
        vec[(root_idx + interval) % 12] = 1.0
    return vec


def parse_chord_name(chord_name: str) -> tuple[str, str]:
    """
    Parse a chord name into (root, quality).
    Examples: 'Am' -> ('A', 'min'), 'Cmaj7' -> ('C', 'maj7'), 'F#' -> ('F#', 'maj')
    """
    if len(chord_name) >= 2 and chord_name[1] in '#b':
        root = chord_name[:2]
        quality_str = chord_name[2:]
    else:
        root = chord_name[0]
        quality_str = chord_name[1:]

    # Map quality strings to our internal types
    quality_map = {
        '': 'maj',
        'm': 'min', 'min': 'min',
        'dim': 'dim', '°': 'dim',
        'aug': 'aug', '+': 'aug',
        '7': '7',
        'maj7': 'maj7', 'M7': 'maj7', 'Δ7': 'maj7',
        'min7': 'min7', 'm7': 'min7',
        'dim7': 'dim7', '°7': 'dim7',
        'm7b5': 'm7b5', 'ø': 'm7b5',
        'sus2': 'sus2',
        'sus4': 'sus4',
        '6': '6',
        'min6': 'min6', 'm6': 'min6',
        '9': '9',
        'min9': 'min9', 'm9': 'min9',
        'maj9': 'maj9', 'M9': 'maj9',
        '11': '11',
        '13': '13',
        'add9': 'add9',
        '5': '5',
    }

    quality = quality_map.get(quality_str, 'maj')
    return root, quality


def chord_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two chord vectors."""
    import math
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_key_from_chords(chords: list[str]) -> str | None:
    """Best-guess the key from a list of chord names."""
    if not chords:
        return None

    scores = {}
    for key, diatonic in DIATONIC_MAJOR.items():
        score = 0
        for chord in chords:
            root, _ = parse_chord_name(chord)
            # Match exact root against diatonic chord roots
            diatonic_roots = [parse_chord_name(d)[0] for d in diatonic]
            if root in diatonic_roots:
                score += 1
        # Bonus: if I chord (first diatonic) appears, extra weight
        tonic = parse_chord_name(diatonic[0])[0]
        tonic_count = sum(1 for c in chords if parse_chord_name(c)[0] == tonic)
        score += tonic_count * 0.5
        scores[key] = score

    return max(scores, key=scores.get) if scores else None
