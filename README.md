# On-Device Elastic Dubbing Engine (youtube-dubber-v3)

A production-grade, offline-first automated video dubbing pipeline. The system translates video dialog from English to localized Hinglish, generates high-fidelity TTS vocal stems, and dynamically compiles a synchronized virtual timeline. By extending video frames rather than warping or compressing speech signals, the engine preserves natural conversational flow without pitch or cadence distortion.

---

## 🚀 Key Architectural Features

1. **Dual-Environment Process Isolation:**
   Segregates heavy neural processing (`faster-whisper`, `demucs`, PyTorch) into an isolated worker sandbox (`.venv_ingest`) separate from the main pipeline orchestrator (`.venv`). This eliminates runtime library conflicts (e.g., NumPy binary ABI incompatibilities) and makes it ready to deploy on system environments with restricted dependencies.

2. **Parametric Dynamic Sidechain Ducking:**
   Instead of using static, global background attenuation that creates an unnatural "sonic vacuum" during speech pauses, the engine splits vocal playback to act as a dynamic compression key:
   ```text
   [Ambiance Track Source] ──► [Sidechain Compress: 1.0kHz - 3.5kHz Formant Band] ──┐
                                                     ▲                              ▼
                                      (Vocal Trigger Sidechain)              [Audio Summer] ──► Output
                                                     │                              ▲
   [Local Voice Stream Input] ───────────────────────┴──────────────────────────────┘
   ```
   This compresses the background room tone *only* during active speech segments, leaving ambient transitions natural.

3. **Overlap-Add (OLA) Phonetic Windowing:**
   Conjoins subsequent speech segments using an $80\text{ms}$ Hann-like (`qsin` envelope) crossfade to prevent sharp phonetic boundaries. For silence gaps, the engine automatically bypasses crossfades and performs direct concatenation to avoid fading ambient background noises into speech boundaries.

4. **Bidirectional Virtual Time Coordinate Transformer:**
   Dynamic frameholds expand the total video runtime. The data contract exposes a serialized coordinate mapping matrix (`virtual_timeline_map`), which maps the expanded $T_{\text{virtual}}$ coordinates back to read-only container timeline coordinates ($T_{\text{physical}}$), facilitating scrubber seek interception in under $180\text{ms}$ on mobile media players (e.g., ExoPlayer).

---

## 🛠️ System Architecture Diagram

```
[ Input: raw_video.mp4 ]
         │
         ▼
[ main.py (System Orchestrator) ]
         │
         ├──► [Phase 1: ingestion_pass.py] (Isolated Subprocess in .venv_ingest)
         │         ├── Demucs (Vocal / Accompaniment separation)
         │         └── Faster-Whisper (ASR segment boundaries & word-level parsing)
         │
         ├──► [Phase 2: inference_queue.py] (Pipeline Control in .venv)
         │         ├── Structured translation lookahead via Local Ollama (gemma4:e2b)
         │         ├── Local neural TTS synthesis (Piper ONNX Engine)
         │         └── Manifest compilation & drift matrix calculation
         │
         └──► [Phase 3: player_emulator.py] (Rendering Engine in .venv)
                   └── Monolithic O(1) multi-stream FFmpeg viewport matrix compilation
```

---

## 📦 Environment Setup & Dependencies

The system requires two localized virtual environments to ensure complete process and dependency separation.

### System Prereqs
* **FFmpeg 8.x** (with `swresample` and `libx264` libraries installed)
* **Ollama** running locally with the target translation model pulled:
  ```bash
  ollama pull gemma4:e2b
  ```

### 1. Main Orchestration Environment (`.venv`)
Install the lightweight orchestration and TTS packages:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Ensure piper-tts is installed inside this environment
pip install piper-tts
```
*Expected dependencies inside `.venv`:* `instructor`, `openai`, `pydantic`, `piper-tts`, `onnxruntime`.

### 2. Ingestion Worker Environment (`.venv_ingest`)
Setup the heavy neural compute environment:
```bash
python3 -m venv .venv_ingest
source .venv_ingest/bin/activate
pip install torch Demucs faster-whisper pathvalidate
```
*Expected dependencies inside `.venv_ingest`:* `torch`, `demucs`, `faster-whisper`.

---

## 🚦 Usage Guide

The pipeline runs end-to-end via the main orchestrator CLI:

```bash
# Activate the main environment
source .venv/bin/activate

# Execute the dubbing pipeline
python main.py <input_video.mp4> -o <output_video.mp4>
```

### CLI Options:
* `input_video`: Path to the source MP4 file to dub.
* `-o`, `--output`: Path to write the final dub-mixed video container (default: `final_elastic_lecture.mp4`).

---

## 🔍 Under the Hood: Pipeline Phase Details

### Phase 1: Ingestion Pass
* Separates background audio (`no_vocals.wav`) from the raw video container using **Demucs**.
* Runs **Faster-Whisper** to extract timestamped dialogue events.
* Intercalates ambient silence gap nodes into the timeline dynamically (threshold: $200\text{ms}$).
* Leverages SHA-256 fingerprinting to cache results and prevent duplicate transcriptions on identical inputs.

### Phase 2: Inference & Synthesis Queue
* Bootstraps Hindi neural voice models (`hi_IN-pratham-medium.onnx`) directly from HuggingFace on first run.
* Loops through the ingestion timeline:
  * Translates dialogue to target Hinglish using Ollama (`gemma4:e2b`), injecting lookahead contexts (`next_context`) to preserve sentence-to-sentence transition coherence.
  * Synthesizes audio segments via the Piper TTS ONNX engine.
  * Measures speech duration ratio ($S_r = D_{\text{nat}} / D_{\text{win}}$) to assign state directives (`NORMAL_SYNC`, `PAD_EMPTY`, `FREEZE_HOLD`).
* Compiles the structural timeline manifest (`timeline_manifest_{hash}.json`).

### Phase 3: Player Emulator Core
* Reconstructs the video matrix by splicing, freezing, and padding the raw video track matching the manifest's structural timeline commands.
* Normalizes all vocal audio tracks to **Stereo 44100Hz** dynamically using FFmpeg's `aresample` filter to ensure perfect sample-rate alignment.
* Blends dialogue transitions via OLA `acrossfade`, applies dynamic sidechain compression to the ambient track, and runs EBU R128 (`loudnorm`) normalization on the master mix.

---

## 📄 Licensing & Manifest Structure

System manifestations, sync maps, and file metadata are saved under the `./studio_assets` directory. The master manifest contains:
* `source_video_hash`: Integrity token mapping the manifest back to the raw source file.
* `playback_timeline`: A list of sequential event nodes carrying timestamps, texts, assigned directives, and local voice chunk locations.
* `virtual_timeline_map`: A pre-computed seek table utilized by downstream media player integrations.
