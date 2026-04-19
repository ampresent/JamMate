#!/usr/bin/env python3
"""
JamMate Standards Pipeline
Download → Analyze → Simulate → Score → Push (one at a time)

Usage: python3 process_standards.py [--start N] [--end N] [--dry-run]
"""

import sys
import os
import json
import math
import wave
import subprocess
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from jammate.standards_db import JAZZ_STANDARDS
from jammate.theory import (
    NOTE_NAMES, CHORD_TYPES, DIATONIC_MAJOR,
    parse_chord_name, build_chord_vector, chord_similarity,
    get_key_from_chords, note_to_index
)
from jammate.synth import AccompanimentGenerator
from jammate.chord_recognition import detect_chords_from_audio

REPO_DIR = Path(__file__).parent.parent
OUTPUT_DIR = REPO_DIR / "analysis"
AUDIO_DIR = REPO_DIR / "audio"
REPORT_FILE = REPO_DIR / "STANDARDS_REPORT.json"


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)


# ─── Analysis ───────────────────────────────────────────────────────────────

def analyze_standard(std):
    """Deep harmonic analysis of a jazz standard."""
    prog = std["progression"]

    # 1. Key detection
    detected_key = get_key_from_chords(prog)

    # 2. Chord complexity score (0-100)
    complexity_scores = []
    for chord in prog:
        root, quality = parse_chord_name(chord)
        quality_to_score = {
            'maj': 10, 'min': 10,
            '5': 15,
            'sus2': 25, 'sus4': 25, 'dim': 25, 'aug': 25,
            '7': 30,
            'maj7': 35, 'min7': 35, 'min6': 35, '6': 35,
            'dim7': 45, 'm7b5': 45,
            '9': 55, 'min9': 55, 'maj9': 55, 'add9': 50,
            '11': 65, '13': 65,
        }
        base_score = quality_to_score.get(quality, 20)
        complexity_scores.append(base_score)
    avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0

    # 3. Chord diversity
    unique_chords = list(dict.fromkeys(prog))
    chord_diversity = len(unique_chords) / len(prog) if prog else 0

    # 4. II-V-I detection (check semitone intervals between roots)
    iivi_count = 0
    for i in range(len(prog) - 2):
        c1_root, c1_qual = parse_chord_name(prog[i])
        c2_root, c2_qual = parse_chord_name(prog[i+1])
        c3_root, c3_qual = parse_chord_name(prog[i+2])

        i1 = note_to_index(c1_root)
        i2 = note_to_index(c2_root)
        i3 = note_to_index(c3_root)

        # ii-V-I patterns: roots form 0→+5→+7 or 0→+7→+0 etc
        c1_to_c2 = (i2 - i1) % 12  # should be 5 (up a fourth) or 7 (up a fifth)
        c2_to_c3 = (i3 - i2) % 12  # should be 5 or 7

        is_iivi = False
        # ii → V → I (e.g. Dm7 → G7 → C): up 5, up 5
        if c1_to_c2 == 5 and c2_to_c3 == 5:
            is_iivi = True
        # ii → V → I with octave wrap
        if c1_to_c2 == 5 and c2_to_c3 == 5:
            is_iivi = True

        # Also check quality: min7 → 7 → maj7/min
        if is_iivi:
            if (c1_qual in ('min7', 'm7b5') and
                c2_qual in ('7',) and
                c3_qual in ('maj7', 'min7', 'maj', 'min')):
                iivi_count += 1

    # Also count "extended ii-V" chains
    for i in range(len(prog) - 1):
        r1, q1 = parse_chord_name(prog[i])
        r2, q2 = parse_chord_name(prog[i+1])
        if q1 == 'min7' and q2 == '7':
            # Check if it's a ii-V movement (root up a fourth)
            if (note_to_index(r2) - note_to_index(r1)) % 12 == 5:
                iivi_count += 0.5  # partial count for ii-V without resolution

    iivi_count = int(iivi_count)

    # 5. Tritone substitution detection
    tritone_subs = 0
    for i in range(len(prog) - 1):
        r1, q1 = parse_chord_name(prog[i])
        r2, q2 = parse_chord_name(prog[i+1])
        if q1 == '7' and q2 in ('maj7', 'min7', 'maj', 'min'):
            idx1 = note_to_index(r1)
            idx2 = note_to_index(r2)
            if abs(idx1 - idx2) == 6:
                tritone_subs += 1

    # 6. Tonal range
    roots_used = set()
    for chord in prog:
        root, _ = parse_chord_name(chord)
        roots_used.add(note_to_index(root))
    tonal_range = len(roots_used)

    # 7. Extension usage
    ext_types = {'9', '11', '13', 'maj9', 'min9', 'add9'}
    extensions = sum(1 for c in prog if parse_chord_name(c)[1] in ext_types)
    ext_ratio = extensions / len(prog) if prog else 0

    # 8. Cadential strength
    cadence_score = 0
    for i in range(len(prog) - 1):
        r1, q1 = parse_chord_name(prog[i])
        r2, q2 = parse_chord_name(prog[i+1])
        if q1 == '7' and q2 in ('maj7', 'min7', 'maj', 'min'):
            idx1 = note_to_index(r1)
            idx2 = note_to_index(r2)
            if (idx1 - idx2) % 12 == 5:
                cadence_score += 1
    cadence_strength = cadence_score / (len(prog) - 1) if len(prog) > 1 else 0

    # 9. Chromatic movement detection
    chromatic_moves = 0
    for i in range(len(prog) - 1):
        r1 = note_to_index(parse_chord_name(prog[i])[0])
        r2 = note_to_index(parse_chord_name(prog[i+1])[0])
        diff = abs(r1 - r2)
        if diff == 1 or diff == 11:
            chromatic_moves += 1

    return {
        "detected_key": detected_key,
        "chord_count": len(prog),
        "unique_chords": len(unique_chords),
        "chord_diversity": round(chord_diversity, 3),
        "avg_complexity": round(avg_complexity, 1),
        "iivi_count": iivi_count,
        "tritone_subs": tritone_subs,
        "tonal_range": tonal_range,
        "extensions_used": extensions,
        "extension_ratio": round(ext_ratio, 3),
        "cadence_strength": round(cadence_strength, 3),
        "chromatic_moves": chromatic_moves,
        "unique_chord_list": unique_chords,
    }


# ─── Simulation ─────────────────────────────────────────────────────────────

def simulate_standard(std):
    """Generate audio and simulate chord recognition."""
    prog = std["progression"]
    style = std.get("style", "jazz")
    tempo = std.get("tempo", 120)

    synth = AccompanimentGenerator(tempo=tempo)
    audio = synth.generate_progression(prog, style)

    # Normalize audio
    peak = np.max(np.abs(audio)) if len(audio) > 0 else 1
    if peak > 0:
        audio = audio / peak * 0.9

    # Save WAV
    title_slug = std["title"].lower().replace(" ", "_").replace("'", "").replace('"', '')
    wav_path = AUDIO_DIR / f"{std['id']:03d}_{title_slug}.wav"
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(synth.sr)
        wf.writeframes(pcm.tobytes())

    # Run chord recognition on the generated audio
    detected = detect_chords_from_audio(audio, synth.sr, window_sec=1.0, hop_sec=0.5)

    # Compare detected vs expected
    expected_chords = prog
    detected_chords = [d["chord"] for d in detected]

    # Simple matching: for each expected chord, see if any detected chord nearby matches
    match_count = 0
    total_compared = min(len(expected_chords), len(detected_chords))

    # Chunk detected chords by bar (approximate)
    bars_per_chord = max(1, len(detected_chords) // len(expected_chords))
    chunked_detected = []
    for i in range(0, len(detected_chords), bars_per_chord):
        chunk = detected_chords[i:i+bars_per_chord]
        if chunk:
            # Most common chord in this chunk
            from collections import Counter
            most_common = Counter(chunk).most_common(1)[0][0]
            chunked_detected.append(most_common)

    for i in range(min(len(expected_chords), len(chunked_detected))):
        exp_root, exp_qual = parse_chord_name(expected_chords[i])
        det_root, det_qual = parse_chord_name(chunked_detected[i])
        if exp_root == det_root:
            match_count += 1

    recognition_accuracy = match_count / len(expected_chords) if expected_chords else 0

    return {
        "wav_path": str(wav_path.relative_to(REPO_DIR)),
        "audio_duration_sec": round(len(audio) / synth.sr, 2),
        "detected_chord_count": len(detected),
        "chunked_detected": chunked_detected[:16],  # First 16 for display
        "recognition_accuracy": round(recognition_accuracy, 3),
        "recognition_match_count": match_count,
        "recognition_total": len(expected_chords),
    }


# ─── Scoring ────────────────────────────────────────────────────────────────

def score_standard(std, analysis, simulation):
    """Score a jazz standard on multiple dimensions (0-100 each)."""

    # 1. Harmonic Richness (complexity + diversity + extensions)
    richness = min(100, (
        analysis["avg_complexity"] * 0.8 +
        analysis["chord_diversity"] * 100 * 0.3 +
        analysis["extension_ratio"] * 250 * 0.15 +
        15  # baseline for jazz chords (7ths, maj7, min7 etc)
    ))

    # 2. Structural Coherence (II-V-I usage, cadences, tonal range)
    iivi_bonus = min(35, analysis["iivi_count"] * 7)
    cadence_bonus = analysis["cadence_strength"] * 25
    range_bonus = min(15, analysis["tonal_range"] * 2)
    chromatic_bonus = min(10, analysis["chromatic_moves"] * 3)
    coherence = min(100, iivi_bonus + cadence_bonus + range_bonus + chromatic_bonus + 15)

    # 3. Playability — synthetic audio is ground truth, so we measure
    #    if the progression is "jammable" (good harmonic rhythm, clear changes)
    #    rather than chord recognition accuracy (which is limited by our synth)
    raw_acc = simulation["recognition_accuracy"]
    chord_change_clarity = min(1.0, analysis["chord_diversity"] * 2)
    harmonic_momentum = min(1.0, (analysis["cadence_strength"] + analysis["iivi_count"] * 0.05))
    playability = min(100, chord_change_clarity * 40 + harmonic_momentum * 40 + raw_acc * 20 + 10)

    # 4. Jazz Authenticity (specific patterns)
    authenticity = 40  # base
    if analysis["iivi_count"] >= 3:
        authenticity += 20
    elif analysis["iivi_count"] >= 1:
        authenticity += 10
    if analysis["tritone_subs"] > 0:
        authenticity += 10
    if analysis["chord_diversity"] > 0.25:
        authenticity += 10
    if analysis["extensions_used"] > 0:
        authenticity += 10
    if analysis["cadence_strength"] > 0.2:
        authenticity += 10
    authenticity = min(100, authenticity)

    # 5. Overall JamMate Score (weighted average)
    overall = (
        richness * 0.25 +
        coherence * 0.30 +
        playability * 0.20 +
        authenticity * 0.25
    )

    return {
        "harmonic_richness": round(richness, 1),
        "structural_coherence": round(coherence, 1),
        "playability": round(playability, 1),
        "jazz_authenticity": round(authenticity, 1),
        "overall_score": round(overall, 1),
        "grade": (
            "S" if overall >= 90 else
            "A" if overall >= 80 else
            "B" if overall >= 70 else
            "C" if overall >= 60 else
            "D" if overall >= 50 else "F"
        ),
    }


# ─── Git Push ───────────────────────────────────────────────────────────────

def git_push(std, commit_msg):
    """Stage results and push to repo."""
    askpass = REPO_DIR.parent / ".git_askpass.sh"
    env = os.environ.copy()
    env["GIT_ASKPASS"] = str(askpass)

    cmds = [
        ["git", "add", "-A"],
        ["git", "commit", "-m", commit_msg],
        ["git", "push"],
    ]

    for cmd in cmds:
        result = subprocess.run(
            cmd, cwd=str(REPO_DIR), env=env,
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0 and "nothing to commit" not in result.stderr:
            print(f"  ⚠ Git warning: {result.stderr.strip()[:200]}")


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def process_one(std, dry_run=False):
    """Process a single jazz standard through the full pipeline."""
    title = std["title"]
    std_id = std["id"]

    print(f"\n{'='*60}")
    print(f"  [{std_id:3d}/100] {title}")
    print(f"  Composer: {std['composer']} | Key: {std['key']} | Tempo: {std['tempo']} BPM")
    print(f"{'='*60}")

    # Phase 1: Analyze
    print(f"  🔍 Analyzing...")
    analysis = analyze_standard(std)
    print(f"     Key: {analysis['detected_key']} | Chords: {analysis['chord_count']} | "
          f"Unique: {analysis['unique_chords']} | II-V-I: {analysis['iivi_count']}")

    # Phase 2: Simulate
    print(f"  🎹 Simulating...")
    simulation = simulate_standard(std)
    print(f"     Audio: {simulation['audio_duration_sec']}s | "
          f"Recognition: {simulation['recognition_accuracy']*100:.0f}%")

    # Phase 3: Score
    print(f"  📊 Scoring...")
    score = score_standard(std, analysis, simulation)
    print(f"     Richness: {score['harmonic_richness']:.0f} | "
          f"Coherence: {score['structural_coherence']:.0f} | "
          f"Playability: {score['playability']:.0f} | "
          f"Authenticity: {score['jazz_authenticity']:.0f}")
    print(f"     ⭐ Overall: {score['overall_score']:.0f} ({score['grade']})")

    # Build result record
    result = {
        "id": std_id,
        "title": title,
        "composer": std["composer"],
        "key": std["key"],
        "tempo": std["tempo"],
        "style": std["style"],
        "form": std["form"],
        "bars": std["bars"],
        "progression": std["progression"],
        "analysis": analysis,
        "simulation": {
            "wav_path": simulation["wav_path"],
            "audio_duration_sec": simulation["audio_duration_sec"],
            "recognition_accuracy": simulation["recognition_accuracy"],
        },
        "score": score,
    }

    # Save individual result
    title_slug = title.lower().replace(" ", "_").replace("'", "").replace('"', '')
    result_path = OUTPUT_DIR / f"{std_id:03d}_{title_slug}.json"
    if not dry_run:
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Phase 4: Push
    if not dry_run:
        print(f"  📤 Pushing to repo...")
        commit_msg = f"feat(standards): {title} — Score {score['overall_score']:.0f} ({score['grade']})"
        try:
            git_push(std, commit_msg)
            print(f"  ✅ Pushed!")
        except Exception as e:
            print(f"  ⚠ Push failed: {e}")

    return result


def main():
    dry_run = "--dry-run" in sys.argv
    start_id = 1
    end_id = 100

    for arg in sys.argv[1:]:
        if arg.startswith("--start="):
            start_id = int(arg.split("=")[1])
        if arg.startswith("--end="):
            end_id = int(arg.split("=")[1])

    ensure_dirs()

    # Filter standards to process
    to_process = [s for s in JAZZ_STANDARDS if start_id <= s["id"] <= end_id]
    to_process.sort(key=lambda s: s["id"])

    print(f"\n🎸 JamMate Standards Pipeline")
    print(f"   Processing {len(to_process)} standards (IDs {start_id}-{end_id})")
    print(f"   Dry run: {dry_run}")
    print()

    all_results = []
    start_time = time.time()

    for std in to_process:
        try:
            result = process_one(std, dry_run=dry_run)
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ Error processing {std['title']}: {e}")
            import traceback
            traceback.print_exc()

    # Update master report
    elapsed = time.time() - start_time
    if not dry_run:
        # Load existing report if any
        existing = []
        if REPORT_FILE.exists():
            with open(REPORT_FILE) as f:
                existing = json.load(f)

        # Merge (replace any existing entries with same id)
        existing_ids = {r["id"] for r in all_results}
        merged = [r for r in existing if r["id"] not in existing_ids] + all_results
        merged.sort(key=lambda r: r["id"])

        with open(REPORT_FILE, 'w') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        # Final push with report
        try:
            askpass = REPO_DIR.parent / ".git_askpass.sh"
            env = os.environ.copy()
            env["GIT_ASKPASS"] = str(askpass)
            subprocess.run(["git", "add", "-A"], cwd=str(REPO_DIR), env=env, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"docs(standards): Master report — {len(merged)} standards analyzed"],
                cwd=str(REPO_DIR), env=env, capture_output=True
            )
            subprocess.run(["git", "push"], cwd=str(REPO_DIR), env=env, capture_output=True)
            print(f"\n  📋 Master report pushed!")
        except Exception as e:
            print(f"\n  ⚠ Report push failed: {e}")

    # Summary
    if all_results:
        scores = [r["score"]["overall_score"] for r in all_results]
        grades = [r["score"]["grade"] for r in all_results]
        from collections import Counter
        grade_dist = Counter(grades)

        print(f"\n{'='*60}")
        print(f"  📊 PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"  Processed: {len(all_results)} standards")
        print(f"  Time: {elapsed:.1f}s ({elapsed/len(all_results):.1f}s each)")
        print(f"  Avg Score: {sum(scores)/len(scores):.1f}")
        print(f"  Best: {max(scores):.1f} ({max(all_results, key=lambda r: r['score']['overall_score'])['title']})")
        print(f"  Worst: {min(scores):.1f} ({min(all_results, key=lambda r: r['score']['overall_score'])['title']})")
        print(f"  Grades: {dict(grade_dist)}")
        print()

        # Top 10
        top10 = sorted(all_results, key=lambda r: r["score"]["overall_score"], reverse=True)[:10]
        print(f"  🏆 Top 10:")
        for i, r in enumerate(top10):
            print(f"     {i+1}. {r['title']} — {r['score']['overall_score']:.0f} ({r['score']['grade']})")


if __name__ == '__main__':
    main()
