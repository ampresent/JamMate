"""
Terminal UI for JamMate.
Shows live chord detection, progression, and jam status.
"""

import sys
import time


# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RED = "\033[31m"
WHITE = "\033[37m"


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def print_header():
    print(f"""{BOLD}{CYAN}
  ╔══════════════════════════════════════╗
  ║          🎸  JamMate  🎸             ║
  ║     AI Jamming Partner               ║
  ╚══════════════════════════════════════╝{RESET}
""")


def print_status(chord: str, confidence: float, history: list[str],
                  predicted: list[str], key: str = None,
                  style: str = "jazz", bpm: int = 120):
    """Print current jam status (single line update)."""
    # Chord display
    conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
    conf_color = GREEN if confidence > 0.7 else YELLOW if confidence > 0.5 else RED

    line = f"  {BOLD}{GREEN}♪ {chord:8s}{RESET} "
    line += f"{conf_color}{conf_bar}{RESET} {confidence:.0%}"
    line += f"  {DIM}({style} @ {bpm}bpm){RESET}"

    if key:
        line += f"  {CYAN}Key: {key}{RESET}"

    sys.stdout.write(f"\r{' ' * 100}\r{line}")
    sys.stdout.flush()


def print_history(history: list[str], max_display: int = 8):
    """Print chord history."""
    display = history[-max_display:]
    print(f"\n  {DIM}History:{RESET} {' → '.join(display)}")


def print_predictions(predicted: list[str]):
    """Print predicted chords."""
    pred_str = ' → '.join(predicted)
    print(f"  {MAGENTA}🔮 Next:{RESET} {pred_str}")


def print_help():
    """Print help text."""
    print(f"""
  {BOLD}Controls:{RESET}
    {CYAN}Ctrl+C{RESET}  Stop jamming
    {CYAN}q{RESET}       Quit

  {BOLD}Modes:{RESET}
    {GREEN}live{RESET}    Real-time microphone input
    {YELLOW}file{RESET}    Analyze an audio file

  {BOLD}Styles:{RESET}
    jazz, blues, rock, pop, funk
""")


def print_session_summary(session):
    """Print end-of-session summary."""
    print(f"\n{BOLD}{CYAN}  Session Summary{RESET}")
    print(f"  {'─' * 35}")
    print(f"  Duration:  {time.time() - session.start_time:.0f}s")
    print(f"  Chords:    {len(session.chord_history)}")
    print(f"  Style:     {session.style}")
    print(f"  BPM:       {session.bpm}")

    if session.chord_history:
        chords = session.chords_only
        unique = list(dict.fromkeys(chords))  # preserve order, deduplicate
        print(f"  Unique:    {', '.join(unique)}")

    key = session.detected_key
    if key:
        print(f"  Est. Key:  {key}")

    print()
