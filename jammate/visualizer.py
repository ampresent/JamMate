"""
Progression visualizer — renders chord progressions as ASCII diagrams.
"""

from .theory import parse_chord_name, NOTE_NAMES, DIATONIC_MAJOR


# Roman numeral representations by scale degree
ROMAN_NUMERALS_MAJOR = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']
ROMAN_NUMERALS_MINOR = ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII']


def chord_to_roman(chord_name: str, key: str = 'C') -> str:
    """Convert a chord name to its roman numeral in a given key."""
    root, quality = parse_chord_name(chord_name)

    diatonic = DIATONIC_MAJOR.get(key, DIATONIC_MAJOR['C'])
    diatonic_roots = [parse_chord_name(d)[0] for d in diatonic]

    if root in diatonic_roots:
        degree = diatonic_roots.index(root)
        numeral = ROMAN_NUMERALS_MAJOR[degree]
        # Add quality extensions
        if quality == '7':
            numeral += '7'
        elif quality == 'maj7':
            numeral += 'maj7'
        elif quality == 'min7':
            numeral += '7'
        elif quality == 'm7b5':
            numeral += 'ø7'
        return numeral

    return chord_name  # Can't romanize, return as-is


def render_progression_bar(chord: str, width: int = 8,
                            highlight: bool = False) -> str:
    """Render a single chord as a bar block."""
    padding = max(0, width - len(chord) - 2)
    left = padding // 2
    right = padding - left
    inner = f"{' ' * left}{chord}{' ' * right}"

    if highlight:
        return f"┃{inner}┃"
    else:
        return f"│{inner}│"


def render_progression(chords: list[str], key: str = None,
                        show_roman: bool = True,
                        current_idx: int = -1) -> str:
    """
    Render a chord progression as an ASCII diagram.

    Example output:
    │  Dm7  │  G7   │ Cmaj7 │  Am7  │
    │  ii   │  V    │   I   │  vi   │
    """
    if not chords:
        return ""

    bar_width = 8
    lines = []

    # Top border
    lines.append("┌" + "┬".join(["─" * bar_width] * len(chords)) + "┐")

    # Chord names
    chord_row = ""
    for i, chord in enumerate(chords):
        chord_row += render_progression_bar(chord, bar_width,
                                             highlight=(i == current_idx))
    lines.append(chord_row)

    # Roman numerals
    if show_roman and key:
        roman_row = "│"
        for chord in chords:
            rn = chord_to_roman(chord, key)
            padding = max(0, bar_width - len(rn) - 2)
            left = padding // 2
            right = padding - left
            roman_row += f" {' ' * left}{rn}{' ' * right} │"
        lines.append(roman_row)

    # Bottom border
    lines.append("└" + "┴".join(["─" * bar_width] * len(chords)) + "┘")

    return "\n".join(lines)


def render_jam_status(chord: str, confidence: float,
                       history: list[str], predicted: list[str],
                       key: str = None, beat: int = 0) -> str:
    """Render a compact jam status display."""
    lines = []

    # Current chord with confidence
    conf_bar_len = int(confidence * 20)
    conf_bar = "█" * conf_bar_len + "░" * (20 - conf_bar_len)
    lines.append(f"  ♪ {chord:8s}  [{conf_bar}] {confidence:.0%}")

    # Key
    if key:
        lines.append(f"  Key: {key}")

    # Progression visualization
    if history:
        display = history[-8:]
        lines.append("")
        lines.append("  History:")
        lines.append(render_progression(display, key))

    # Predicted
    if predicted:
        lines.append("")
        lines.append("  Predicted:")
        lines.append(render_progression(predicted, key))

    return "\n".join(lines)


def print_chord_wheel(key: str = 'C'):
    """Print a chord wheel showing all diatonic chords in a key."""
    diatonic = DIATONIC_MAJOR.get(key, DIATONIC_MAJOR['C'])

    print(f"\n  Chord Wheel — Key of {key}")
    print(f"  {'─' * 40}")

    for i, (chord, numeral) in enumerate(zip(diatonic, ROMAN_NUMERALS_MAJOR)):
        marker = " ◀ tonic" if i == 0 else ""
        print(f"    {numeral:6s}  {chord:8s}{marker}")

    print()
