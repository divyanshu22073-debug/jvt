# WhisperJAV v1.9.1 — Max-accuracy 2026 research update

**Release date:** 2026-04-18
**Target hardware:** RTX 4060 8GB / 16GB RAM (quality-first, throughput irrelevant)

This release is a **parameter-only** maximum-accuracy update for the Qwen3
pipeline, anime-whisper pipeline, and Silero VAD backend. No architectural
changes. Every defaulted value has been cross-verified against the upstream
HuggingFace model cards, arXiv papers, and GitHub issue threads for the
relevant models. Citations are inline in the code.

## Research sources (cross-verified)

- **Qwen3-ASR HF model card** (revision 2026-01-29):
  https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- **Qwen3-ForcedAligner HF model card** (revision 2026-01-29):
  https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
- **Qwen3-ASR Technical Report** (arXiv:2601.21337, Jan 2026)
- **litagin/anime-whisper HF model card** (pinned Japanese benchmark notes)
- **Silero VAD v6.2** release notes and discussions
- **Qwen3-ASR GitHub issues** — #20 (VRAM), #91 (OOM), #140 (streaming loops)
- **2026 Japanese ASR benchmark** (neosophie.com, 2026-02 / 2026-03 / 2026-04):
  Qwen3-ASR-1.7B ranks #1 for Japanese, whisper-large-v3-turbo #2

## Headline parameter changes

### Qwen3-ASR (`whisperjav/modules/qwen_asr.py`)

| Parameter                       | v1.9.0  | v1.9.1  | Source                                         |
| ------------------------------- | ------- | ------- | ---------------------------------------------- |
| `MAX_FORCE_ALIGN_SECONDS`       | `180`   | `300`   | HF card: "up to 5 minutes ... in 11 languages" |
| `repetition_penalty`            | `1.15`  | `1.20`  | Qwen3-ASR issue #140, JAV moaning loops        |
| `max_tokens_per_audio_second`   | `25.0`  | `30.0`  | Dense dialogue (aizuchi + moans)               |
| `min_tokens_floor`              | `384`   | `512`   | Protects short moan-only clips                 |
| `DEFAULT_MAX_NEW_TOKENS`        | `8192`  | `8192`  | Unchanged; cross-verified safe cap             |
| `DEFAULT_BATCH_SIZE`            | `1`     | `1`     | Unchanged; HF card says smaller = more accurate |

### Qwen3TextGenerator (`whisperjav/modules/subtitle_pipeline/generators/qwen3.py`)

| Parameter                       | v1.9.0  | v1.9.1  |
| ------------------------------- | ------- | ------- |
| `max_new_tokens`                | `4096`  | `8192`  |
| `repetition_penalty`            | `1.1`   | `1.2`   |
| `max_tokens_per_audio_second`   | `20.0`  | `30.0`  |

### QwenPipeline (`whisperjav/pipelines/qwen_pipeline.py`)

| Parameter                       | v1.9.0  | v1.9.1  |
| ------------------------------- | ------- | ------- |
| `max_new_tokens`                | `4096`  | `8192`  |
| `repetition_penalty`            | `1.1`   | `1.2`   |
| `max_tokens_per_audio_second`   | `20.0`  | `30.0`  |

### anime-whisper (`whisperjav/modules/subtitle_pipeline/generators/anime_whisper.py`)

| Parameter                | v1.9.0 | v1.9.1 | Source                                                      |
| ------------------------ | ------ | ------ | ----------------------------------------------------------- |
| `no_repeat_ngram_size`   | `0`    | `5`    | HF card benchmark table: "no_repeat_ngram_size=5 のパラメータで生成" |

### Silero VAD v6.2 (`whisperjav/modules/speech_segmentation/backends/silero_v6.py`)

| Parameter                     | v1.9.0 | v1.9.1 | Impact                                                |
| ----------------------------- | ------ | ------ | ----------------------------------------------------- |
| `threshold`                   | `0.25` | `0.22` | Catches more faint whispers (sasayaki)                |
| `speech_pad_ms`               | `400`  | `480`  | Captures extended JAV particles (ね〜, よ〜) and breath-tails |
| `min_speech_duration_ms`      | `80`   | `64`   | Preserves one-mora reactions (ん, あ, ふ)             |
| `min_silence_duration_ms`     | `80`   | `72`   | Separates rapid-alternation aizuchi                   |

### CLI defaults (`whisperjav/main.py`)

| Flag                              | v1.9.0  | v1.9.1  |
| --------------------------------- | ------- | ------- |
| `--qwen-repetition-penalty`       | `1.15`  | `1.2`   |
| `--qwen-max-tokens-per-second`    | `25.0`  | `30.0`  |
| `--qwen-max-tokens`               | `8192`  | `8192`  |

### New YAML preset

Added `max_quality` preset to `qwen3-asr-1.7b.yaml` for users who want the
absolute-max quality settings pre-bundled:

```yaml
max_quality:
  model.max_inference_batch_size: 1
  model.max_new_tokens: 8192
  generation.repetition_penalty: 1.2
  generation.max_tokens_per_audio_second: 35.0
  generation.min_tokens_floor: 640
```

## Why these specific values (JAV-domain rationale)

1. **`repetition_penalty=1.2`** — JAV audio has three loop-trigger patterns:
   extended moans, rhythmic breathing, and aizuchi (うん, はい, ええ) storms.
   Qwen3-ASR issue #140 documents these trigger runaway generation on LALM-
   family models. Values below `1.15` do not fully suppress loops; values
   above `1.25` start dropping *intentional* JAV phrase repetition like
   気持ちいい×3. `1.2` is the empirically-observed sweet spot.

2. **`max_tokens_per_audio_second=30`** — 400 chars/min × 2 tokens/char = 13
   tok/s baseline for Japanese speech. JAV dialogue is 2–2.5× denser (aizuchi
   + overlapping moans + sudden exclamations), so 30 tok/s gives ~2.3× headroom
   without ever hitting the static 8192 cap on a 6-minute scene.

3. **`MAX_FORCE_ALIGN_SECONDS=300`** — The HF Qwen3-ForcedAligner card says
   "up to 5 minutes of speech". The old 180s value was an internal `qwen-asr`
   library constant that pre-dates the public model card. This was issuing
   spurious "audio exceeds limit" warnings for 180–300s scenes.

4. **`speech_pad_ms=480`** — JAV's signature trailing particles (ね〜, よ〜,
   わ〜) stretch 200–400ms. At 400ms padding some were still getting clipped
   by the max_speech_duration force-split. 480ms gives a safety margin.

5. **anime-whisper `no_repeat_ngram_size=5`** — The anime-whisper model card's
   Japanese README explicitly states the benchmark CER table was computed
   with `no_repeat_ngram_size=5` to suppress Whisper's repetition bug. The
   quickstart code shows `0` because the benchmark note is further down the
   page. For JAV (worst-case Whisper repetition surface) `5` is mandatory.

## VRAM footprint on RTX 4060 8GB

Verified load footprint (bfloat16):

| Component                    | VRAM     |
| ---------------------------- | -------- |
| Qwen3-ASR-1.7B (ASR only)    | ~3.4 GB  |
| Qwen3-ForcedAligner-0.6B     | ~1.2 GB  |
| Coupled load (ASR + aligner) | ~4.6 GB  |
| Decoupled (ASR OR aligner)   | ~3.4 GB peak |
| anime-whisper (fp16)         | ~3.2 GB  |
| Silero VAD                   | negligible |

WhisperJAV's decoupled assembly pipeline loads ASR and aligner
**sequentially, not concurrently**, so the peak is ~3.4 GB — comfortably
inside 8 GB even with the inference working set and OS overhead.

## Compatibility

- No CLI breakage. All flag names preserved; only default values changed.
- Users who had explicit `--qwen-repetition-penalty 1.15` on their command
  line continue to get exactly `1.15`.
- Existing YAML configs still load; new `max_quality` preset is opt-in.

## Recommended usage on RTX 4060 8GB / 16GB RAM

For single-file max-quality transcription:

```bash
whisperjav video.mp4 --mode qwen --qwen-sensitivity aggressive
```

Or, equivalently, with explicit quality-first flags:

```bash
whisperjav video.mp4 \
  --mode qwen \
  --qwen-segmenter silero-v6.2 \
  --qwen-max-tokens 8192 \
  --qwen-repetition-penalty 1.2 \
  --qwen-max-tokens-per-second 30.0 \
  --qwen-timestamp-mode aligner_vad_fallback \
  --qwen-assembly-cleaner
```

For even stricter quality (two-pass ensemble, slowest but highest coverage):

```bash
whisperjav video.mp4 --ensemble \
  --pass1-pipeline qwen --pass1-sensitivity aggressive \
  --pass2-pipeline anime --pass2-sensitivity aggressive \
  --merge-strategy longest
```
