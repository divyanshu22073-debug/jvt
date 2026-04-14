#!/usr/bin/env python3
"""Version information for WhisperJAV."""

# PEP 440 compliant version for pip/wheel
__version__ = "1.9.0"

# Human-readable version for display in UI
__version_display__ = "1.9.0"

# Version metadata
__version_info__ = {
    "major": 1,
    "minor": 9,
    "patch": 0,
    "release": "",
    "architecture": "v5.0"
}

# =============================================================================
# What's New in v1.9.0 (Architecture v5.0)
# =============================================================================
# - Max-accuracy defaults: aggressive sensitivity, Silero v6.2 VAD tuned for
#   maximum Japanese line coverage (threshold=0.25, speech_pad_ms=400)
# - ASR model upgrades:
#   * Faster-Whisper default: large-v3-turbo (6x faster, near-identical WER)
#   * Whisper default: large-v3 (best multilingual accuracy)
#   * Kotoba-Whisper: v2.2 (latest Japanese-optimized, Mar 2026)
#   * Qwen3-ASR: 1.7B with improved token budgets and alignment
#   * Transformers: kotoba-whisper-v2.2 default
# - VAD improvements:
#   * Silero v6.2 tuned for max speech capture (threshold 0.25)
#   * Increased speech padding to 400ms for Japanese trailing particles
#   * Reduced min_silence_duration_ms to 80ms for detecting short pauses
# - Translation provider updates:
#   * Gemini: gemini-2.5-flash (latest)
#   * Claude: claude-sonnet-4-20250514 (latest)
#   * GPT: gpt-4.1-mini (latest)
#   * Groq: llama-4-scout-17b-16e-instruct (latest)
# - Post-processing improvements:
#   * Better hallucination detection with 2026 research methods
#   * Improved Japanese repetition cleaning
#   * Enhanced sentence boundary detection
# - Dependencies updated to latest stable versions
# =============================================================================
