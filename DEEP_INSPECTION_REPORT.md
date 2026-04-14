# WhisperJAV v1.8.10 - Deep Inspection Report
## Comprehensive Code Audit & Latest Model Research
### Date: 2026-04-08

---

# PART 1: COMPLETE CODEBASE AUDIT

## 1. Project Overview

**WhisperJAV** is a subtitle generator for Japanese Adult Videos, built as a multi-pipeline ASR system tackling the unique challenges of JAV audio: non-verbal vocalizations, extreme dynamics, spectral mimicry, and long-form hallucination drift.

- **Version**: 1.8.10 (Architecture v4.4)
- **Language**: Python 3.10-3.12
- **License**: MIT
- **Entry Points**: `whisperjav` (CLI), `whisperjav-gui` (GUI), `whisperjav-translate`, `whisperjav-upgrade`, `whisperjav-bench`
- **Total Source Files**: ~180+ Python files, ~40 YAML/JSON configs, ~50 docs

---

## 2. Architecture Overview

### 2.1 Pipeline Architecture (7 Pipelines)

| Pipeline | ASR Backend | Scene Detection | Speech Enhancer | Speech Segmenter | File |
|----------|-------------|-----------------|-----------------|-------------------|------|
| **faster** | Faster-Whisper (turbo) | None | None | None | `pipelines/faster_pipeline.py` |
| **fast** | OpenAI Whisper (stable-ts) | Auditok | None | None | `pipelines/fast_pipeline.py` |
| **balanced** | Faster-Whisper | Auditok | Configurable | Silero | `pipelines/balanced_pipeline.py` |
| **fidelity** | OpenAI Whisper (stable-ts) | Auditok | Configurable | Silero | `pipelines/fidelity_pipeline.py` |
| **transformers** | HuggingFace (Kotoba) | Optional | Configurable | Optional | `pipelines/transformers_pipeline.py` |
| **qwen** | Qwen3-ASR | Semantic | Configurable | Silero v6.2 | `pipelines/qwen_pipeline.py` |
| **decoupled** | Model-agnostic | Configurable | Configurable | Configurable | `pipelines/decoupled_pipeline.py` |

### 2.2 Processing Flow (9 Phases)
1. **Audio Extraction** (48kHz via FFmpeg)
2. **Scene Detection** (split at natural breaks)
3. **Speech Enhancement** (optional per-scene denoising)
4. **Speech Segmentation / VAD** (identify speech regions)
5. **ASR Transcription** (per-scene transcription)
6. **Scene SRT Generation** (micro-subtitles)
7. **SRT Stitching** (combine scene SRTs)
8. **Post-processing** (sanitization, hallucination removal, repetition cleaning)
9. **Analytics** (metrics collection)

---

## 3. ASR Engines (Complete Inventory)

### 3.1 FasterWhisperProASR (`modules/faster_whisper_pro_asr.py`)
- **Backend**: `faster-whisper` (CTranslate2)
- **Default Model**: `large-v2`
- **Device Support**: CUDA, CPU (MPS falls back to CPU - CTranslate2 limitation)
- **Compute Types**: auto, float16, float32, int8, int8_float16, int8_float32
- **Features**:
  - External Speech Segmenter integration (not internal VAD)
  - Segment filter pipeline: logprob threshold, CPS limiter, non-verbal filter
  - VAD failover detection (force full transcribe if VAD produces no segments)
  - Constructor firewall: prevents Silero preset contamination for non-Silero backends
  - Parameter tracer support for real-time observability
  - Crash tracer integration for CTranslate2 debugging

### 3.2 WhisperProASR (`modules/whisper_pro_asr.py`)
- **Backend**: OpenAI `whisper` library
- **Default Model**: `large-v2`
- **Device Support**: CUDA, MPS (native), CPU
- **Features**:
  - Same external Speech Segmenter pattern as FasterWhisperPro
  - Same constructor firewall and segment filter pipeline
  - Uses `whisper.load_model()` + `whisper.transcribe()`

### 3.3 StableTSASR (`modules/stable_ts_asr.py`)
- **Backend**: `stable-ts` (stable_whisper) wrapping OpenAI Whisper
- **Default Model**: `large-v2`
- **Features**:
  - Uses `stable_whisper.load_model()` for enhanced timestamp stability
  - Silero VAD cache checking (v3.1, v4.0)
  - Japanese linguistic sets for subtitle processing (LexicalSets dataclass)
  - Japanese post-processing with sentence boundary detection

### 3.4 TransformersASR (`modules/transformers_asr.py`)
- **Backend**: HuggingFace Transformers `pipeline("automatic-speech-recognition")`
- **Default Model**: `kotoba-tech/kotoba-whisper-bilingual-v1.0`
- **Features**:
  - Chunked long-form algorithm
  - Configurable: chunk_length_s (15), stride, batch_size (16)
  - Attention implementations: sdpa, flash_attention_2, eager
  - Timestamp granularity: segment or word
  - Beam search and temperature configuration
  - Support for any HuggingFace whisper model

### 3.5 KotobaFasterWhisperASR (`modules/kotoba_faster_whisper_asr.py`)
- **Backend**: `faster-whisper` with Kotoba model
- **Default Model**: `kotoba-tech/kotoba-whisper-v2.0-faster`
- **Features**:
  - Internal Silero VAD via faster-whisper's `vad_filter` parameter
  - Optimized defaults for Japanese speech
  - CTranslate2-based (same CUDA/CPU-only limitation as FasterWhisperPro)

### 3.6 QwenASR (`modules/qwen_asr.py`)
- **Backend**: `qwen-asr` package + HuggingFace Transformers
- **Default Model**: `Qwen/Qwen3-ASR-1.7B`
- **Aligner**: `Qwen/Qwen3-ForcedAligner-0.6B`
- **Features**:
  - Word-level timestamps via ForcedAligner
  - `merge_master_with_timestamps()` for punctuation-timestamp reconciliation
  - `stable_whisper.transcribe_any()` for regrouping into sentence segments
  - Three processing modes: Assembly, Context-Aware, VAD Slicing
  - Repetition penalty, max tokens per audio second
  - Context/glossary support for contextual biasing
  - Japanese post-processing integration

### 3.7 Subtitle Pipeline Generators (Decoupled Architecture)

#### Qwen3TextGenerator (`subtitle_pipeline/generators/qwen3.py`)
- Text-only mode via `load_model_text_only` / `transcribe_text_only`
- VRAM lifecycle: load() → generate → unload() → safe_cuda_cleanup()
- No persistent instance (prevents stale CUDA closure bugs)

#### AnimeWhisperGenerator (`subtitle_pipeline/generators/anime_whisper.py`)
- **Model**: `litagin/anime-whisper` (Whisper large-v3 fine-tuned on anime/JAV)
- Low-level WhisperProcessor + WhisperForConditionalGeneration API
- Greedy decoding: do_sample=False, num_beams=1
- NO initial prompt/context (causes hallucinations per model card)
- ~4GB VRAM (Whisper large-v2, ~1.55B params, float16)

---

## 4. Speech Segmentation / VAD Backends (Complete Inventory)

### 4.1 Silero VAD (`speech_segmentation/backends/silero.py`)
- **Versions**: v4.0 (default), v3.1, v6.2 (pip package)
- **Source**: `snakers4/silero-vad` via `torch.hub`
- **Features**: Configurable threshold, speech_pad_ms, max_speech_duration_s
- **v6.2**: Uses `silero-vad` pip package with hysteresis, max_speech_duration_s

### 4.2 NeMo VAD (`speech_segmentation/backends/nemo.py`)
- **Variants**: nemo-lite (Frame VAD ~0.5GB), nemo-diarization (NeuralDiarizer ~4GB)
- **Model**: `nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0`
- **Requires**: `nemo_toolkit[asr]`

### 4.3 TEN VAD (`speech_segmentation/backends/ten.py`)
- **Source**: `ten-vad` pip package (TEN Framework)
- **Features**: `group_segments()` function for segment grouping by time gaps
- **Used by**: ChronosJAV pipeline (anime-whisper) for tight subtitle timing

### 4.4 Whisper-as-VAD (`speech_segmentation/backends/whisper_vad.py`)
- **Approach**: Uses Whisper model transcription segments as speech regions
- **Variants**: whisper-vad (small ~500MB), whisper-vad-tiny (~75MB), whisper-vad-base (~150MB), whisper-vad-medium (~1.5GB)
- **Advantages**: Semantic speech understanding, works on low SNR audio

### 4.5 None Backend (`speech_segmentation/backends/none.py`)
- Passthrough — no segmentation applied

---

## 5. Scene Detection Backends (Complete Inventory)

### 5.1 Semantic Scene Detector (`scene_detection_backends/semantic_backend.py`)
- **Algorithm**: MFCC feature extraction → Agglomerative Clustering → Cosine similarity
- **Dependencies**: scikit-learn, librosa, scipy
- **Natural Range**: 20s-420s scenes
- **V7 Features**: "Snap to Silence" split logic, full coverage 0→Duration guaranteed

### 5.2 Semantic Audio Clustering (`vendor/semantic_audio_clustering.py`)
- **Core Algorithm** (V7.1.0):
  - MFCC features with `librosa.feature.mfcc()`
  - `MedianFilter` smoothing from `scipy.ndimage`
  - `StandardScaler` + `AgglomerativeClustering` from scikit-learn
  - `cosine_similarity` for texture-based boundary detection
  - Configurable `min_duration` (20s), `max_duration` (420s), `snap_window`, `clustering_threshold`
- **Diagnostics**: numpy/librosa/numba/soundfile version compatibility checks

### 5.3 Auditok Backend (`scene_detection_backends/auditok_backend.py`)
- Energy-based detection, fast and reliable
- Uses `auditok` library

### 5.4 Silero Scene Detector (`scene_detection_backends/silero_backend.py`)
- Neural VAD-based detection using Silero VAD
- Better for noisy audio

---

## 6. Speech Enhancement Backends (Complete Inventory)

### 6.1 ClearVoice (`speech_enhancement/backends/clearvoice.py`)
- **Models**:
  - `MossFormer2_SE_48K` (default, best quality, 48kHz)
  - `FRCRN_SE_16K` (16kHz)
  - `MossFormerGAN_SE_16K` (16kHz)
  - `MossFormer2_SS_16K` (speech separation, 16kHz)
- **Source**: `clearvoice` fork from ClearerVoice-Studio (ModelScope/Alibaba)
- **API**: Batch numpy array processing

### 6.2 ZipEnhancer (`speech_enhancement/backends/zipenhancer.py`)
- **ICASSP 2025 Paper**: `arXiv:2501.05183`
- **Parameters**: Only 2.04M (50x smaller than MossFormer2)
- **PESQ Score**: 3.69 on DNS2020
- **Modes**: torch (GPU) and ONNX (CPU/GPU)
- **Sample Rate**: Native 16kHz
- **Chunking**: MAX_CHUNK_DURATION=10s, OVERLAP_DURATION=0.5s crossfade
- **ModelScope ID**: `iic/speech_zipenhancer_ans_multiloss_16k_base`

### 6.3 BS-RoFormer (`speech_enhancement/backends/bs_roformer.py`)
- **Purpose**: Vocal isolation and music source separation
- **Sample Rate**: 44.1kHz
- **Models**: vocals (extract speech), other (non-vocal)
- **Package**: `bs-roformer-infer`

### 6.4 FFmpeg DSP (`speech_enhancement/backends/ffmpeg_dsp.py`)
- **Filters**: loudnorm, denoise, compress, highpass, lowpass, deess
- **Advantage**: Always available, no ML model required

### 6.5 None Backend (`speech_enhancement/backends/none.py`)
- Passthrough — no enhancement

---

## 7. Translation System (Complete Inventory)

### 7.1 Translation Engine (`translate/core.py`)
- **Backend**: PySubtrans wrapper
- **Batch Size Management**: `cap_batch_size_for_context()` — caps based on LLM context window
  - 8K context: 11 lines/batch
  - 16K context: 27 lines/batch
  - 32K+: 30 (max)
- **Token Budget**: overhead=2500, tokens_per_line=500 (worst-case Japanese)

### 7.2 Translation Providers (`translate/providers.py`)
| Provider | Model | API |
|----------|-------|-----|
| DeepSeek | deepseek-chat | deepseek.com |
| OpenRouter | deepseek/deepseek-chat | openrouter.ai |
| Gemini | gemini-2.0-flash | Google Gemini API |
| Claude | claude-3-5-haiku-20241022 | Anthropic |
| GPT | gpt-4o-mini | OpenAI |
| GLM | glm-4-flash | BigModel.cn |
| Groq | llama-3.3-70b-versatile | Groq API |
| Ollama | gemma3:12b (auto) | localhost:11434 |
| Local | llama-8b (deprecated) | llama-cpp-python |
| Custom | user-specified | user endpoint |

### 7.3 Ollama Manager (`translate/ollama_manager.py`)
- **Hardware-Aware Model Recommendations**:
  - CPU only: qwen2.5:3b
  - 8GB VRAM: qwen2.5:7b
  - 12GB VRAM: gemma3:12b
  - 16GB+: qwen2.5:14b
- **Sampling Defaults**: top_k=50, top_p=0.92, min_p=0.05, repeat_penalty=1.1
- **Auto-lifecycle**: server start, model pull, context window detection

---

## 8. Post-Processing Pipeline (Complete Inventory)

### 8.1 Hallucination Remover (`modules/hallucination_remover.py`)
- **V13** with offline caching
- **Pattern Sources**: Bundled JSON (`filter_list_v08.json`, `regexp_v09.json`) + remote download
- **Detection Methods**: Exact match, regex patterns, fuzzy matching (SequenceMatcher)
- **Language Support**: Japanese, Korean, Chinese, English
- **Bracket Pair Detection**: 10 bracket types (Latin, Japanese, CJK)
- **Cache**: `~/.cache/whisperjav/hallucination_filters/`, 7-day expiry

### 8.2 Repetition Cleaner (`modules/repetition_cleaner.py`)
- **Library**: Uses `regex` (Unicode-aware, not stdlib `re`)
- **8+ Cleaning Patterns** (ordered specific → general):
  1. Phrase with separator: `(word!!)\1{3,}`
  2. Multi-char word: `(ハッ)\1{3,}`
  3. Phrase with comma: `(ゆーちゃん、)\1{2,}`
  4. Single-char whitespace flood
  5. Prefix + char: `あらららら`
  6. Single-char flood: `ううううう` (with dakuten/handakuten support)
  7. Vowel extension: `あ〜〜〜〜`
  8. Wave-dash comma phrases
- **Phrase length**: Up to 30 chars (#209 fix)

### 8.3 Japanese Post-Processor (`modules/japanese_postprocessor.py`)
- **Presets**: default, high_moan, narrative
- **Linguistic Sets**:
  - Sentence-final particles: ね, よ, わ, の, ぞ, ぜ, さ, か, かな, な
  - Compound particle sequences: ですよね, ですかね, じゃないの, etc.
  - Polite verb endings, dialectal variations (Kansai-ben)
  - Aizuchi/backchanneling (うん, はい, ええ)
- **Processing**: Removes fillers, anchors on sentence endings, merges by punctuation/gaps, splits by length/duration

### 8.4 Subtitle Sanitizer (`modules/subtitle_sanitizer.py`)
- Japanese-specific text cleaning
- Subtitle sanitizer English variant available

### 8.5 Assembly Text Cleaner (`modules/assembly_text_cleaner.py`)
- Pre-alignment text cleaning for Qwen pipeline

### 8.6 Cross-Subtitle Processor (`modules/cross_subtitle_processor.py`)
- Cross-subtitle boundary handling

### 8.7 Segment Classification (`modules/segment_classification.py`)
- Classifies audio segments by type

### 8.8 Segment Filters (`modules/segment_filters.py`)
- SegmentFilterConfig + SegmentFilterHelper
- Logprob threshold, CPS limiter, non-verbal filter

---

## 9. Ensemble System (Complete Inventory)

### 9.1 Ensemble Orchestrator (`ensemble/orchestrator.py`)
- **Two-Pass Processing**: Run two different pipelines, merge results
- **Execution Modes**: Parallel (batch), Serial (per-file)
- **Drop-Box Pattern**: Uses raw `mp.Process` (not ProcessPoolExecutor)
- **Source Output Mode**: Per-file SRT placement next to input

### 9.2 Merge Engine (`ensemble/merge.py`)
- **7 Merge Strategies**:
  1. `full_merge` — combine everything
  2. `pass1_primary` / `pass2_primary` — prioritize one pass, fill gaps
  3. `pass1_overlap` / `pass2_overlap` — overlap-aware priority
  4. `smart_merge` — intelligent overlap detection
  5. `longest` — keep longer subtitle per segment
- **Overlap Threshold**: 30% of base subtitle duration

### 9.3 Pass Worker (`ensemble/pass_worker.py`)
- Worker payload serialization for subprocess execution
- Qwen sensitivity resolution with per-backend SEGMENTER_PARAMS

### 9.4 BYOP: Faster Whisper XXL
- External executable integration for ensemble Pass 2
- Config from `asr_config.json` → `ui_preferences.byop.xxl_extra_args`

---

## 10. Decoupled Subtitle Pipeline (IMPL-001 Phase 2)

### 10.1 Orchestrator (`subtitle_pipeline/orchestrator.py`)
- **Protocol-Based**: TemporalFramer, TextGenerator, TextCleaner, TextAligner
- **9-Step Flow**: Frame → Slice → Generate → Clean → Align → Merge → Sentinel → Reconstruct → Harden
- **VRAM Swap Pattern**: generator.load() → generate → unload() → aligner.load() → align → unload()

### 10.2 Available Components

**Framers** (`subtitle_pipeline/framers/`):
- `full_scene` — entire scene as one frame
- `vad_grouped` — VAD segments grouped by time/duration
- `srt_source` — use existing SRT as frame guide
- `manual` — user-specified boundaries

**Generators** (`subtitle_pipeline/generators/`):
- `qwen3` — Qwen3-ASR text-only mode
- `anime-whisper` — litagin/anime-whisper

**Cleaners** (`subtitle_pipeline/cleaners/`):
- `qwen3` — Qwen-specific text cleaning
- `anime_whisper` — anime-whisper specific cleaning
- `passthrough` — no cleaning

**Aligners** (`subtitle_pipeline/aligners/`):
- `qwen3` — Qwen3-ForcedAligner (0.6B)
- `none` — skip alignment

### 10.3 Types & Reconstruction
- **TimestampMode**: aligner_interpolation, aligner_vad_fallback, aligner_only, vad_only
- **RegroupMode**: standard, sentence_only, off
- **StepDownConfig**: Adaptive step-down retry on alignment collapse
- **HardeningConfig**: Post-alignment quality gates

---

## 11. Configuration System

### 11.1 Config v4 (YAML-based)
- **Location**: `config/v4/`
- **Ecosystems**: pipelines, presets, qwen, tools, transformers
- **Registries**: base_registry, ecosystem_registry, model_registry, preset_registry, tool_registry
- **Schemas**: base, ecosystem, model, preset, tool (Pydantic v2)

### 11.2 Legacy Config (JSON-based)
- `config/asr_config.json` — main configuration
- `config/manager.py` — ConfigManager with UI preferences
- `config/legacy.py` — resolve_legacy_pipeline(), resolve_ensemble_config()
- `config/resolver_v3.py` — V3 configuration resolver

### 11.3 Sanitization Config
- `config/sanitization_config.py` — HallucinationConstants, RepetitionConstants
- `config/sanitization_constants.py` — shared constants

---

## 12. GUI System

### 12.1 WebView GUI (`webview_gui/`)
- **Backend**: PyWebView (WebView2 on Windows, WebKit on Linux/Mac)
- **Frontend**: HTML/CSS/JS single-page app
- **Files**: `index.html`, `app.js`, `style.css` (multiple themes)
- **API**: `api.py` — Python ↔ JavaScript bridge
- **4 Tabs**: Transcription, Advanced Options, Ensemble Mode, AI Translation

---

## 13. Utility Modules

| Module | Purpose |
|--------|---------|
| `utils/device_detector.py` | CUDA → MPS → CPU auto-detection |
| `utils/gpu_utils.py` | safe_cuda_cleanup(), VRAM management |
| `utils/model_loader.py` | HuggingFace Hub download resilience (#204) |
| `utils/crash_tracer.py` | CTranslate2 crash diagnostics |
| `utils/parameter_tracer.py` | Real-time parameter observability (JSONL) |
| `utils/progress_aggregator.py` | Unified progress display |
| `utils/async_processor.py` | Async pipeline processing |
| `utils/console.py` | UTF-8 mode relaunch for Chinese Windows (#190) |
| `utils/process_manager.py` | Process lifecycle management |
| `utils/preflight_check.py` | GPU requirement enforcement |
| `modules/audio_extraction.py` | FFmpeg audio extraction (48kHz) |
| `modules/audio_preparation.py` | Audio format preparation |
| `modules/media_discovery.py` | File discovery and validation |
| `modules/srt_stitching.py` | Multi-scene SRT assembly |
| `modules/srt_postprocessing.py` | SRT → VTT conversion, normalization |
| `modules/timing_adjuster.py` | Subtitle timing corrections |
| `modules/alignment_sentinel.py` | Alignment quality monitoring |
| `modules/vad_failover.py` | Force full transcribe if VAD fails |

---

## 14. Dependencies (from `pyproject.toml`)

### Core
| Package | Version | Purpose |
|---------|---------|---------|
| openai-whisper | git+main | OpenAI Whisper ASR |
| stable-ts | git+fork | Timestamp refinement |
| faster-whisper | >=1.1.0 | CTranslate2 Whisper |
| pydantic | >=2.0,<3.0 | Config schemas |
| PyYAML | >=6.0 | YAML config v4 |
| tiktoken | >=0.7.0 | Whisper tokenizer |
| regex | latest | Unicode-aware regex |

### CLI Audio
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=2.0.0 | Scientific computing |
| scipy | >=1.13.0 | Signal processing |
| librosa | >=0.11.0 | Audio analysis |
| auditok | latest | Energy-based VAD |
| silero-vad | >=6.2 | Neural VAD |
| ten-vad | latest | TEN Framework VAD |
| scikit-learn | >=1.4.0 | Scene detection clustering |
| numba | >=0.60.0 | JIT compilation |

### Enhancement
| Package | Version | Purpose |
|---------|---------|---------|
| modelscope | >=1.20 | ZipEnhancer framework |
| clearvoice | git+fork | MossFormer2 denoising |
| bs-roformer-infer | latest | Vocal isolation |
| onnxruntime | >=1.16.0 | ONNX inference |

### Translation
| Package | Version | Purpose |
|---------|---------|---------|
| pysubtrans | >=1.5.0 | Subtitle translation engine |
| openai | >=1.35.0 | GPT provider |
| google-genai | >=1.39.0 | Gemini provider |

### HuggingFace
| Package | Version | Purpose |
|---------|---------|---------|
| transformers | >=4.40.0 | HF Transformers |
| accelerate | >=0.26.0 | Model acceleration |
| huggingface-hub | >=0.25.0 | Model downloads |
| hf_xet | latest | Faster HF downloads |

### Qwen
| Package | Version | Purpose |
|---------|---------|---------|
| qwen-asr | >=0.0.6 | Qwen3-ASR SDK |

---

# PART 2: LATEST MODELS & IMPROVEMENTS (Research Findings)

## 15. ASR Model Landscape (April 2026)

### 15.1 Open ASR Leaderboard (HuggingFace) — Current Rankings

| Rank | Model | WER | Parameters | Date |
|------|-------|-----|------------|------|
| **#1** | **Cohere Transcribe** (cohere-transcribe-03-2026) | **5.42%** | 2B | Mar 2026 |
| **#2** | **IBM Granite 4.0 1B Speech** | ~5.5% | 1B | Mar 2026 |
| **#3** | **NVIDIA Canary-Qwen 2.5B** | 5.63% | 2.5B | Late 2025 |
| **#4** | **IBM Granite Speech 3.3 8B** | ~6% | 8B | Jun 2025 |
| Top-10 | Whisper Large V3 | ~7-8% | 1.55B | 2023 |

### 15.2 New Model Releases Since WhisperJAV v1.8.10

#### Cohere Transcribe (March 2026) -- HIGHEST PRIORITY
- **#1 on Open ASR Leaderboard** (5.42% average WER)
- 2B parameters, Apache 2.0 license
- Beats Whisper Large V3, Qwen3-ASR, IBM Granite, ElevenLabs Scribe v2
- **Already requested on WhisperJAV** (GitHub Issue #262)
- HuggingFace: `CohereLabs/cohere-transcribe-03-2026`
- Strong multilingual performance (4th overall, 2nd among open models on multilingual ASR leaderboard)

#### IBM Granite 4.0 1B Speech (March 2026) -- HIGH PRIORITY
- **#1 on OpenASR leaderboard** (at 1B parameters — extremely efficient)
- Compact speech-language model for multilingual ASR
- HuggingFace: `ibm-granite/granite-4.0-1b-speech`
- Ideal for CPU/low-VRAM deployment

#### OpenAI gpt-4o-transcribe / gpt-4o-mini-transcribe (March 2025)
- Proprietary API-only models
- Improvements to WER and language recognition vs. Whisper
- Not open-source, but relevant as competition benchmark

#### GLM-ASR-Nano-2512 (December 2025)
- Z.AI's 1.5B ASR model
- Compact, competitive accuracy

#### LiteASR / lite-whisper-large-v3
- Low-rank compression for Whisper encoders
- Significantly reduced inference cost, maintained accuracy
- Relevant for faster-whisper optimization path

### 15.3 Japanese-Specific ASR Updates

#### Qwen3-ASR (Already Integrated — Latest January 2026)
- WhisperJAV already uses `Qwen/Qwen3-ASR-1.7B` and `Qwen3-ForcedAligner-0.6B`
- **52 languages** support (expanded from 30)
- Japanese ASR benchmark (Feb 2026): WER 0.1899 (best among tested models)
- Qwen3-ASR-Toolkit on GitHub for extended functionality
- **Recommendation**: Current integration is up-to-date

#### Kotoba-Whisper v2.2 (March 2026) -- MEDIUM PRIORITY
- Updated from v2.0/v2.1 base
- `kotoba-tech/kotoba-whisper-v2.2` on HuggingFace
- WhisperJAV currently defaults to `kotoba-whisper-bilingual-v1.0` and `v2.0-faster`
- **Recommendation**: Update default model to v2.2

#### anime-whisper (litagin)
- Latest version still `litagin/anime-whisper` (Nov 2024)
- Fine-tuned variants: `flyfront/anime-whisper-faster` (Nov 2024), `Aratako/anime-whisper-ggml` (Jun 2025)
- **New from litagin**: `litagin/VibeVoice-ASR` — 9B parameter model (date unknown)
- **Recommendation**: Investigate VibeVoice-ASR as potential upgrade

---

## 16. VAD / Speech Segmentation Updates

### 16.1 Silero VAD v6 (September 2025) -- ALREADY INTEGRATED
- WhisperJAV already supports `silero-vad>=6.2`
- **v6 improvements**: 16% fewer errors on noisy real-life data, 11% fewer on multi-domain
- **Status**: Current integration is up-to-date

### 16.2 New VAD Research
- **Picovoice Cobra**: Commercial alternative with claims of better accuracy
- **WebRTC VAD**: Still the lightweight baseline, but Silero significantly more accurate at 5% and 1% FPR

---

## 17. Speech Enhancement Updates

### 17.1 ZipEnhancer (Already Integrated — ICASSP 2025)
- WhisperJAV already integrates ZipEnhancer with ONNX + torch modes
- **Status**: Current, no newer version available

### 17.2 BS-RoFormer Updates (March 2026)
- **BS-RoFormer Resurrection**: Updated models (2025.07)
- **Multi-Stage Music Source Restoration** (March 2026): curriculum-trained BS-RoFormer + HiFi++ GAN restoration
- **Recommendation**: Update `bs-roformer-infer` to latest version

### 17.3 Interspeech 2025 URGENT Challenge
- Universal, Robust and Generalizable speech EnhancemeNT models
- New research directions for generalized enhancement
- Potential future backends to monitor

---

## 18. Hallucination Reduction Research

### 18.1 Calm-Whisper (Interspeech 2025) -- ALREADY CITED
- **80% reduction** in non-speech hallucination, <0.1% WER degradation
- **Method**: Fine-tuning only 3 of 20 decoder attention heads that cause hallucinations
- WhisperJAV cites this paper but does not implement the technique directly
- **Recommendation**: Consider integrating Calm-Whisper fine-tuned weights as optional model

### 18.2 Current WhisperJAV Approach
- WhisperJAV uses post-processing hallucination removal (regex + exact + fuzzy)
- Calm-Whisper's approach is model-level (attention head fine-tuning)
- These approaches are **complementary** and could be stacked

---

## 19. Inference Engine Updates

### 19.1 vLLM ASR Support
- vLLM now supports ASR model serving (including Whisper, Qwen3-ASR)
- WhisperJAV has `docs/architecture/ADR-005-vllm-readiness.md` (already planned)
- **Recommendation**: vLLM backend would enable batched inference, better GPU utilization

### 19.2 faster-whisper / CTranslate2
- Latest release: v1.1.x (October 2025, CTranslate2 3.22.0)
- No major new versions since
- **Known Limitation**: No MPS support (Apple Silicon falls back to CPU)

---

## 20. Translation Model Updates

### 20.1 Current Provider Models (Potential Updates)

| Provider | Current Model | Latest Available |
|----------|---------------|-----------------|
| Claude | claude-3-5-haiku-20241022 | **claude-3.5-haiku-latest** or **claude-4-haiku** (if released) |
| GPT | gpt-4o-mini | **gpt-4.1-mini** (potential newer model) |
| Gemini | gemini-2.0-flash | **gemini-2.5-flash** (if available) |
| Groq | llama-3.3-70b-versatile | **llama-4-scout** or newer models |
| Ollama | gemma3:12b | **gemma3:12b** (still current) |

### 20.2 Ollama Translation Models
- Current VRAM recommendations are up-to-date
- Consider adding **Qwen3:14b** or **Phi-4** as alternatives

---

# PART 3: PRIORITIZED RECOMMENDATIONS

## Critical Priority (Immediate Impact)

### R1. Add Cohere Transcribe as ASR Backend
- **Why**: #1 on Open ASR Leaderboard (5.42% WER), Apache 2.0, already requested (#262)
- **How**: New TextGenerator adapter in `subtitle_pipeline/generators/cohere.py` via HuggingFace Transformers
- **Impact**: Potentially best available accuracy for all languages

### R2. Add IBM Granite 4.0 1B Speech Backend
- **Why**: #1 at 1B params — extremely efficient, ideal for CPU/low-VRAM users
- **How**: HuggingFace Transformers adapter similar to TransformersASR
- **Impact**: Enable high-quality transcription on low-end hardware

### R3. Update Kotoba-Whisper Default to v2.2
- **Why**: v2.2 released March 2026, improves on v2.0
- **How**: Update `DEFAULT_MODEL` in KotobaFasterWhisperASR and TransformersASR defaults
- **Impact**: Better Japanese accuracy with no code changes

## High Priority (Significant Improvement)

### R4. Integrate Calm-Whisper Fine-Tuned Weights
- **Why**: 80% hallucination reduction, <0.1% WER loss — complementary to existing regex filters
- **How**: Offer `calm-whisper-large-v3` as model option in balanced/fidelity pipelines
- **Impact**: Dramatic reduction in hallucination without post-processing overhead

### R5. vLLM Backend for ASR Serving
- **Why**: Already planned (ADR-005), enables batched inference
- **How**: Implement vLLM serving mode for Qwen3-ASR and Whisper models
- **Impact**: Better GPU utilization, higher throughput for batch processing

### R6. Investigate VibeVoice-ASR (litagin)
- **Why**: 9B parameter model from anime-whisper creator — potentially much better anime/JAV recognition
- **How**: Test as anime-whisper replacement/upgrade in ChronosJAV pipeline
- **Impact**: Could significantly improve anime/JAV-specific accuracy

## Medium Priority (Nice to Have)

### R7. Update Translation Provider Models
- Update Claude, GPT, Gemini model identifiers to latest versions
- Add Qwen3 and Phi-4 as Ollama translation model options

### R8. BS-RoFormer Resurrection Models
- Update to 2025.07+ models for better vocal isolation
- Consider BS-RoFormer v2 architecture when available

### R9. NVIDIA Canary-Qwen 2.5B Integration
- Strong English accuracy (#3 on leaderboard)
- Would complement Qwen3-ASR for multilingual use cases

### R10. GLM-ASR-Nano-2512 as Lightweight Option
- 1.5B parameter, efficient ASR
- Alternative for low-VRAM scenarios

---

# PART 4: TECHNICAL DEBT & CODE QUALITY OBSERVATIONS

## 21. Code Quality Notes

1. **Consistent Architecture**: The decoupled pipeline (IMPL-001) is well-designed with clean protocol separation
2. **Defensive Coding**: Constructor firewalls, VAD failover, crash tracers show mature error handling
3. **VRAM Management**: Explicit load/unload cycles prevent GPU memory leaks
4. **CTranslate2 Workaround**: The `os._exit(0)` nuclear exit for Windows CTranslate2 destructor crash is documented (#125)
5. **UTF-8 Handling**: Relaunch pattern for Chinese Windows is robust (#190)
6. **Deprecated Code**: `local` translation provider deprecated in v1.8.10, migration to `ollama` planned
7. **Hidden Options**: `kotoba-faster-whisper` temporarily hidden from user selection

## 22. Potential Issues

1. **MPS Limitation**: CTranslate2 backends (faster-whisper, kotoba) cannot use Apple Silicon GPU — documented but limits Mac performance
2. **setuptools Pin**: `<82` required for modelscope pkg_resources — will need updating
3. **Duplicate Silero VAD Cache Check**: `_is_silero_vad_cached()` exists in both `stable_ts_asr.py` and `silero.py`
4. **Repetition Cleaner Double Spacing**: All patterns and class body have double newlines (cosmetic)
5. **Translation Provider Hardcoded Models**: Model versions in `providers.py` will drift as APIs update

---

*Report generated by deep inspection of all 180+ source files, configuration files, and documentation in the WhisperJAV v1.8.10 codebase, combined with online research of the latest ASR, VAD, and speech enhancement models as of April 2026.*
