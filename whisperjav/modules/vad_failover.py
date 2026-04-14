"""Utility helpers for deciding when to bypass VAD and transcribe the full clip.

v1.9.0: Lowered min_duration_for_fallback and improved coverage detection
to ensure maximum line coverage even when VAD is too conservative.
"""
from __future__ import annotations

from typing import Iterable, List, Dict


def _flatten_segments(vad_segments: Iterable[Iterable[Dict[str, float]]]) -> List[Dict[str, float]]:
    """Return a flat list of VAD timestamp dictionaries."""
    flat: List[Dict[str, float]] = []
    for group in vad_segments or []:
        flat.extend(group or [])
    return flat


def _speech_coverage_seconds(flat_segments: List[Dict[str, float]]) -> float:
    """Compute total speech duration from flattened VAD segments."""
    total = 0.0
    for seg in flat_segments:
        start = float(seg.get("start_sec", 0.0))
        end = float(seg.get("end_sec", 0.0))
        if end > start:
            total += end - start
    return total


def should_force_full_transcribe(
    vad_segments: Iterable[Iterable[Dict[str, float]]],
    audio_duration: float,
    min_duration_for_fallback: float = 60.0,  # v1.9.0: lowered from 120s to catch more failures
    min_coverage_ratio: float = 0.005,  # v1.9.0: lowered from 0.01 for aggressive coverage
) -> bool:
    """Decide whether VAD output looks obviously wrong and warrants a full-clip pass.

    v1.9.0: More aggressive fallback thresholds to ensure maximum line coverage.
    The heuristic keeps the original behaviour for short clips while catching the
    failure mode reported by users where long scenes (>1 minute) end up with only
    a handful of subtitles or none at all because VAD returned almost no speech.
    """
    if audio_duration <= 0:
        return False

    if audio_duration < min_duration_for_fallback:
        return False

    flat_segments = _flatten_segments(vad_segments)
    if not flat_segments:
        return True

    speech_seconds = _speech_coverage_seconds(flat_segments)
    coverage_ratio = speech_seconds / audio_duration if audio_duration else 0.0
    if coverage_ratio < min_coverage_ratio:
        return True

    # If VAD only detected a single tiny blob on a very long clip, treat it as failure.
    if len(flat_segments) <= 2 and audio_duration >= (min_duration_for_fallback * 3):
        return True

    return False
