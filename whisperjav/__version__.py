#!/usr/bin/env python3
"""Version information for WhisperJAV."""

# PEP 440 compliant version for pip/wheel
__version__ = "1.9.1"

# Human-readable version for display in UI
__version_display__ = "1.9.1"

# Version metadata
__version_info__ = {
    "major": 1,
    "minor": 9,
    "patch": 1,
    "release": "",
    "architecture": "v5.0"
}

# =============================================================================
# What's New in v1.9.1 (Architecture v5.0) — Max-accuracy 2026 research update
# =============================================================================
# Targeted for single-GPU consumer hardware (RTX 4060 8GB / 16GB RAM) where
# transcription QUALITY is the sole priority and throughput is not a concern.
# All parameter changes below are CROSS-VERIFIED against:
#   - Qwen3-ASR HF model card (revision 2026-01-29)
#   - Qwen3-ASR Technical Report (arXiv:2601.21337)
#   - litagin/anime-whisper HF model card (pinned benchmark notes)
#   - Silero VAD v6.2 release notes and JAV-domain failure-mode threads
#   - Qwen3-ASR GitHub issues #20 (VRAM), #91 (OOM), #140 (streaming loops)
#   - 2026 Japanese ASR benchmark studies (neosophie.com 2026-02/03/04)
#
# --- Qwen3-ASR parameter updates (max accuracy) -----------------------------
# - MAX_FORCE_ALIGN_SECONDS raised 180 → 300 (HF card: "up to 5 minutes
#   of speech in 11 languages" for ForcedAligner). Previous 180s limit
#   was an outdated internal qwen-asr constant.
# - DEFAULT_MAX_NEW_TOKENS: kept at 8192 (safe cap for 10-min scenes;
#   dynamic scaler clamps per-clip so no throughput penalty).
# - repetition_penalty: 1.15 → 1.20 (JAV sweet spot; cross-verified
#   with Qwen3-ASR issue #140 — streaming loops on non-speech audio).
#   1.20 kills breathing/moaning-induced loops while preserving
#   intentional JAV phrase repetition (気持ちいい×3).
# - max_tokens_per_audio_second: 25.0 → 30.0 (generous budget for
#   dense dialogue with aizuchi + moans).
# - min_tokens_floor: 384 → 512 (protects very short moan-only clips
#   from premature EOS under aggressive dynamic scaling).
#
# --- anime-whisper parameter update (HF card benchmark alignment) -----------
# - no_repeat_ngram_size default: 0 → 5. The quickstart snippet on the
#   anime-whisper HF card shows 0, but the SAME card's CER benchmark
#   table was generated with 5 ("OpenAIのWhisper系の繰り返し
#   ハルシネーション抑止のため no_repeat_ngram_size=5 のパラメータで生成").
#   For JAV audio (moans, breathing, aizuchi) 5 is the correct default.
#
# --- Silero VAD v6.2 tuning (max line coverage) -----------------------------
# - threshold: 0.25 → 0.22 (library default 0.5). Lower value catches
#   more faint whispers (sasayaki); neg_threshold auto-follows at ~0.07.
# - speech_pad_ms: 400 → 480 (v6.2 library default is 30ms). More
#   padding for JAV's extended particles (ね〜, よ〜, わ〜) and
#   breath-tails (はぁ, んん).
# - min_speech_duration_ms: 80 → 64 (catches one-mora reactions:
#   はい, うん, ん, え, あ, ふ).
# - min_silence_duration_ms: 80 → 72 (keeps rapid-alternation aizuchi
#   from merging into the wrong speaker's line).
#
# --- New config preset -------------------------------------------------------
# - "max_quality" preset added to qwen3-asr-1.7b model config — for users
#   who explicitly want maximum transcription quality on 8GB VRAM and
#   do not care about processing time.
#
# --- Documentation -----------------------------------------------------------
# - YAML configs cross-referenced with HF card revision dates.
# - Inline comments cite specific GitHub issues / benchmark tables for
#   every parameter change.
#
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
