# Deep Technical Analysis: On-Device Elastic Dubbing Engine v3.1
**Document:** SPEC-EDB-V31 vs. Prototype Reality Review  
**Codebase:** `/Users/lolet/Downloads/youtube-dubber-v3`  
**Analyzed:** 2026-05-19

---

## 1. System Overview & What Exists Today

The v3.1 spec describes a **4-module pipeline**. Here is what is actually built vs. what is spec-only:

| Module | Spec Component | File | Status |
|--------|---------------|------|--------|
| Phase 1 | Asynchronous Static Ingestion | `ingestion_pass.py` | ✅ Built (functional) |
| Phase 2 | Synchronous Inference Queue | `inference_queue.py` | ✅ Built (functional) |
| Data Contract | Manifest & Orchestration Math | `app_manifest.py` | ✅ Built (well-designed) |
| Phase 3 | Player Emulator / FFmpeg Render | `player_emulator.py` | ✅ Built (functional) |
| — | OLA Crossfade Windowing | *nowhere* | ❌ Missing entirely |
| — | Virtual Timeline Coordinate Transformer | *nowhere* | ❌ Missing entirely |
| — | Ring Buffer (In-Memory PCM streaming) | *nowhere* | ❌ Missing (Python sim uses disk) |
| — | Pipeline Orchestrator / Entry Point | *nowhere* | ❌ Missing entirely |
| — | Piper model file | `piper_models/` (empty dir) | ❌ Missing asset |

---

## 2. Architecture Flow Analysis

### What the spec says the data flow is:
```
VIDEO ──► [PHASE 1: Demucs + Whisper] ──► ingestion_checkpoint_{hash}.json
                                                     │
                                                     ▼
                               [PHASE 2: LLM Translation + Piper TTS]
                                       ──► vocal_chunk_{N}.wav files
                                       ──► timeline_manifest_{hash}.json
                                                     │
                                                     ▼
                               [PHASE 3: FFmpeg Filter Graph Renderer]
                                       ──► final_output.mp4
```

### What actually exists in the code:
- **Phase 1 → Phase 2:** The handoff is via a JSON file path. `ingestion_pass.py` writes `ingestion_checkpoint_{hash}.json`, and `inference_queue.py` reads it. ✅ Clean.
- **Phase 2 → Phase 3:** `inference_queue.py` writes `timeline_manifest_{hash}.json`. `player_emulator.py` reads it plus the checkpoint (to find `source_video_path` and `no_vocals_wav_path`). ✅ Clean.
- **No master orchestrator exists** — there is no `main.py` or pipeline runner that calls all 3 phases in sequence.

---

## 3. Module-by-Module Deep Dive

### 3.1 `ingestion_pass.py` — Phase 1
**Purpose:** Vocal stem separation + ASR segmentation.

**What it does right:**
- SHA-256 fingerprinting (truncated to 12 chars) for idempotent caching ✅
- `demucs --two-stems vocals` correctly isolates `vocals.wav` / `no_vocals.wav` ✅
- Faster-Whisper with INT8 quantization + VAD filter matches spec intent ✅
- Caches raw ASR results to JSON to skip re-transcription ✅

**Critical Bugs / Gaps:**

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| B1 | **Demucs output path mismatch** | 🔴 Critical | `HTDEMUCS_DIR = os.path.join(STUDIO_ASSETS, "htdemucs")` is created but Demucs is told `-o output_dir` where `output_dir = STUDIO_ASSETS`. Demucs then writes to `STUDIO_ASSETS/htdemucs/{base_name}/vocals.wav`. This is consistent *only* if `STUDIO_ASSETS` is passed as `-o`. Verify this resolves correctly end-to-end. |
| B2 | **`PYTORCH_ENABLE_MPS_FALLBACK=1` on Linux/Android** | 🟡 Medium | This env var is Apple-Silicon-specific. On Android (final target) this is a no-op but harmless. On the desktop prototype it should be fine. |
| B3 | **Whisper model `small.en` is English-only** | 🟡 Medium | Spec says the input lecture is English — this is correct. But model download happens at first run with no feedback to the user (Faster-Whisper auto-downloads). No progress indicator or error path for offline use. |
| B4 | **Gap segments not generated** | 🟠 High | If there is silence between Whisper segments (e.g., 2s–5s gap, then 8s–12s segment), the gap [5s–8s] is completely absent from `raw_segments`. Phase 2 will skip it but Phase 3's `bg_trim` will also skip that slice. The final video will have a jump cut in the background ambient track at those positions. |

---

### 3.2 `inference_queue.py` — Phase 2
**Purpose:** LLM translation + Piper TTS + orchestration directive computation.

**What it does right:**
- Instructor + Ollama integration for structured Devanagari output ✅
- Piper TTS subprocess correctly pipes text via `stdin` ✅
- `ffprobe` duration extraction with `nokey=1` fix ✅
- Orchestration math delegated correctly to `app_manifest.compute_orchestration_directives()` ✅
- Failed segment `continue` guard prevents corrupt manifest nodes ✅

**Critical Bugs / Gaps:**

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| B5 | **Piper model does not exist** | 🔴 Critical | `piper_models/` directory is 34 bytes (likely a `.gitkeep`). The model file `hi_IN-pratham-medium.onnx` + `.onnx.json` config are completely absent. Every TTS call will fail immediately. |
| B6 | **No OLA crossfade context injection** | 🟠 High | Spec §6 explicitly requires the first phrase of segment N+1 to be appended to the LLM prompt for segment N to build phonetic continuity. The current code processes each segment in total isolation — there is zero lookahead context passed into the LLM prompt. |
| B7 | **`asplit=1` is a no-op and incorrect** | 🟡 Medium | In `player_emulator.py`, `asplit=1` is used for non-PAD_EMPTY/non-FREEZE segments. `asplit` with `n=1` creates one output from one input and is functionally a passthrough. This is harmless but semantically wrong — it should just be a direct stream label reference. The real risk: if FFmpeg ever rejects a 1-input `asplit`, the filter graph will crash. |
| B8 | **No segment skipping compensation** | 🟠 High | If a segment fails TTS (B5), `continue` skips it. But `segment_index` values in the remaining nodes will now be non-contiguous (e.g., 0, 1, 3, 4 skipping 2). The player emulator loops over the manifest as given and processes contiguous indices — but the *background audio trim* in `player_emulator.py` uses `start_ms/end_ms` from the manifest, so audio timing still works. However the video concatenation might create gaps or overlaps at the skip point. |
| B9 | **LLM word-count heuristic is unreliable** | 🟡 Medium | `computed_word_target = max(1, int(target_duration_sec * 2.8))` assumes 2.8 Hindi words/second. But Piper's natural speaking rate for `pratham-medium` is closer to 3.5–4 syllables/second. This will frequently overshoot the window and trigger unnecessary `FREEZE_HOLD` states. |
| B10 | **Fallback translation `नमस्ते`** | 🟠 High | On LLM failure, the system falls back to "Namaste" for every segment. This creates a silent-but-wrong dub that is hard to debug. Should instead skip the segment or flag it. |

---

## 3.3 `app_manifest.py` — Data Contract
**Purpose:** Pydantic models + orchestration math.

**What it does right:**
- Clean Pydantic v2 model hierarchy ✅
- `field_validator` for temporal window integrity ✅
- `compute_orchestration_directives()` math is exactly correct per spec ✅
- `serialize_manifest()` and `load_manifest()` symmetry ✅

**Gaps / Improvements:**

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| B11 | **`accumulated_drift_offset_ms` is always 0** | 🟡 Medium | The field exists in the model but is never computed or populated anywhere in `inference_queue.py`. This is the cumulative clock drift tracker (sum of all previous FREEZE_HOLD deltas) which is essential for the Virtual Timeline Coordinate Transformer in §5. |
| B12 | **No `LectureDubbingManifest` hash verification** | 🟡 Medium | `load_manifest()` does not cross-validate `source_video_hash` against the actual file on disk. A stale manifest from a different video version will silently produce wrong output. |

---

## 3.4 `player_emulator.py` — Phase 3
**Purpose:** Renders the final video using FFmpeg complex filter graph.

**What it does right:**
- Non-destructive `trim`/`atrim` + `setpts`/`asetpts` for precise window slicing ✅
- `tpad=stop_mode=clone` for FREEZE_HOLD frame extension ✅
- `apad` for PAD_EMPTY silence insertion ✅
- Parametric EQ biquad filter for mid-range ducking (`equalizer=f=2250:width_type=h:width=1250`) ✅
- `-16 LUFS` loudnorm broadcast standard ✅
- `-crf 22` with `libx264` fast preset ✅

**Critical Bugs / Gaps:**

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| B13 | **`[1:a]atrim` reuse in filter graph** | 🔴 Critical | Every segment in the loop applies `[1:a]atrim=...` to the *same input stream* `[1:a]` (the `no_vocals.wav`). In FFmpeg filter graphs, **a single input stream cannot be read by multiple filter nodes** without first splitting it. The current code never calls `asplit` on `[1:a]`, so FFmpeg will error: `"Input link [1:a] is referenced more than once"` on any video with more than 1 segment. |
| B14 | **No OLA crossfade in FFmpeg graph** | 🟠 High | Spec §6 requires 50–100ms Hann-windowed crossfade between adjacent vocal segments. The current `concat` filter creates hard audio cuts. |
| B15 | **`loudnorm` applied after `amix`** | 🟡 Medium | The `-af loudnorm` is applied as a simple audio filter *after* mapping `[master_audio_track]`. But `[master_audio_track]` is already a `filter_complex` output. Applying `-af` on top of a `filter_complex` mapped stream can conflict in some FFmpeg versions. Should instead inline `loudnorm` inside the `filter_complex` graph. |
| B16 | **`asplit=1` passthrough nodes** | 🟡 Medium | Same as B7 — all NORMAL_SYNC and FREEZE_HOLD vocal segments use `asplit=1` which is a pointless single-output split. |

---

## 4. Missing Architectural Components

These are **spec-required features with zero implementation**:

### 4.1 Pipeline Orchestrator (main.py) — MISSING
No entry point exists that:
1. Accepts a video path
2. Calls Phase 1 → Phase 2 → Phase 3 in sequence
3. Reports progress to the user

### 4.2 OLA (Overlap-Add) Crossfade Engine — MISSING
Spec §6 is unimplemented. The fix involves:
- In `inference_queue.py`: pass segment N+1's first phrase into N's LLM prompt
- In `player_emulator.py`: replace hard `concat` with `acrossfade=d=0.08:c1=hann:c2=hann` between adjacent vocal chunks

### 4.3 Virtual Timeline Coordinate Transformer — MISSING
Spec §5 defines:  
`T_virtual = T_physical + Σ(ΔD_freeze_i)`

This is needed for:
- Correct seek bar mapping in the Android player
- `accumulated_drift_offset_ms` population in the manifest
- Binary search lookup table generation at manifest build time

**In the Python prototype**, the transformer should be computed during Phase 2 and stored in the manifest so the Android player can consume it directly.

### 4.4 In-Memory Ring Buffer — MISSING (by design)
The spec's §4 ring buffer is an Android-native concept (AHardwareBuffer/AudioTrack). The Python prototype intentionally uses disk-based WAV files as a simulation substitute — this is acceptable for the prototype. **However**, the manifest's `local_path` field in `AudioAssetMetadata` will need to be replaced with a memory handle reference in the Android port.

---

## 5. Dependency & Asset Status

| Dependency | Required By | Status |
|-----------|------------|--------|
| `faster-whisper` | Phase 1 | Likely installed (`.venv` exists) |
| `demucs` | Phase 1 | Likely installed (`.venv` exists) |
| `pydantic` | All | Likely installed |
| `instructor` | Phase 2 | Likely installed |
| `openai` | Phase 2 | Likely installed |
| Ollama + `gemma4:e2b` model | Phase 2 | **UNKNOWN — must verify** |
| `piper` binary | Phase 2 | **Unknown — installed in .venv/bin?** |
| `hi_IN-pratham-medium.onnx` | Phase 2 | ❌ **MISSING — model file absent** |
| `hi_IN-pratham-medium.onnx.json` | Phase 2 | ❌ **MISSING — config absent** |
| `ffmpeg` / `ffprobe` | Phase 1, 3 | System-level — assumed present |
| `local_input.mp4` | Phase 1 | ✅ Present (2.3MB) |

---

## 6. Prioritized Action Plan

### 🔴 P0 — Blockers (Pipeline Cannot Run)

| ID | Action | File |
|----|--------|------|
| A1 | **Fix `[1:a]` multi-read bug** — add `asplit=N` on `no_vocals.wav` at the start of the filter graph and use individual split outputs `[bg_0]`, `[bg_1]`... per segment | `player_emulator.py` |
| A2 | **Download Piper model** — fetch `hi_IN-pratham-medium.onnx` + `.onnx.json` from HuggingFace into `piper_models/` | Asset |
| A3 | **Verify Ollama + gemma4 model** — confirm `ollama list` shows `gemma4:e2b` running on `localhost:11434` | Environment |

### 🟠 P1 — High Priority (Correctness Gaps)

| ID | Action | File |
|----|--------|------|
| A4 | **Create `main.py` orchestrator** — single entry point for Phase 1 → 2 → 3 with CLI args | New file |
| A5 | **Implement `accumulated_drift_offset_ms`** — compute running sum of all FREEZE_HOLD deltas in Phase 2 loop | `inference_queue.py` |
| A6 | **Add silence gap segments** — detect inter-segment gaps in Phase 1, generate synthetic `PAD_EMPTY` nodes to cover them | `ingestion_pass.py` |
| A7 | **Add OLA lookahead context** — in Phase 2 loop, pass `next_segment_text` into LLM system prompt | `inference_queue.py` |
| A8 | **Add OLA crossfade to FFmpeg graph** — replace hard `concat` with `acrossfade` between vocal chunks | `player_emulator.py` |

### 🟡 P2 — Medium Priority (Quality & Robustness)

| ID | Action | File |
|----|--------|------|
| A9 | **Fix `asplit=1` → direct label** — remove pointless single-output splits | `player_emulator.py` |
| A10 | **Move `loudnorm` inside filter_complex** — chain after `amix` output, remove `-af` flag | `player_emulator.py` |
| A11 | **Better LLM failure handling** — skip segment instead of injecting "नमस्ते" | `inference_queue.py` |
| A12 | **Add virtual timeline lookup table** — serialize seek table `[{v_ms, virtual_ms}]` into manifest | `inference_queue.py`, `app_manifest.py` |
| A13 | **Calibrate word-rate heuristic** — measure actual Piper TTS rate and update 2.8 constant | `inference_queue.py` |

---

## 7. Architecture Quality Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Spec Completeness | 6/10 | OLA, Virtual Timeline, Ring Buffer unimplemented |
| Data Contract Design | 9/10 | Pydantic manifest is excellent |
| Phase 1 Code Quality | 8/10 | Solid, minor gap-segment issue |
| Phase 2 Code Quality | 6/10 | B5 (model missing) + B6 (no OLA) are significant |
| Phase 3 Code Quality | 5/10 | B13 is a pipeline-breaking FFmpeg bug |
| Pipeline Integration | 4/10 | No orchestrator, no end-to-end runner |
| **Overall** | **6.3/10** | Good foundation, 3 critical bugs to fix before first run |

---

## 8. Recommended Build Order

```
Week 1: Fix the 3 P0 blockers (A1, A2, A3) + create main.py (A4)
         → Goal: Pipeline runs end-to-end for the first time

Week 2: Implement A5, A6, A7 (silence gaps, drift tracking, OLA LLM context)
         → Goal: Timing accuracy and natural dub flow

Week 3: Implement A8, A10 (OLA FFmpeg crossfade, inline loudnorm)
         → Goal: Smooth audio boundaries, broadcast-quality output

Week 4: A9, A11, A12, A13 (polish, seek table, error handling, calibration)
         → Goal: Production-ready Python prototype ready for Android port
```
