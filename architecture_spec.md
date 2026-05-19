# Technical Specification & System Architecture Blueprint v3.2.0
## Project: On-Device Elastic转Dubbing Engine (`youtube-dubber-v3`)
**System Status:** Production Architecture | **Target Environment:** 100% Offline Android Native (Kotlin/Jetpack Media3)

---

## 1. Executive Summary & Production Paradigm Shift

Traditional automated video dubbing systems rely on destructive digital signal processing (DSP) time-stretching algorithms (e.g., WSOLA, Phase Vocoders) to force translated speech into rigid video time slots. This approach introduces severe acoustic distortions, phase artifacts, and unnatural pacing that cause high cognitive fatigue for students during long study sessions.

`youtube-dubber-v3` solves this problem by introducing a non-destructive **Elastic Timeline Engine**. Instead of warping human speech frequencies, the video container is treated as a flexible canvas. The audio is generated at a natural $1.0\times$ human pace, and the system dynamically manipulates video frame progression—freezing visual frames when explanations run long, and injecting silent spaces when they are concise.

To make this architecture viable for mobile hardware, the system is strictly decoupled into an offline data preparation pass and a lightweight runtime playback layer.

---

## 2. Multi-Environment System Segmentation Layout (TRIZ Principle 1)

To prevent thermal throttling, rapid battery drain, and thread starvation on mobile System-on-Chips (SoCs), computing tasks are cleanly divided based on resource weight:



### 2.1. Parent Process Architecture (`.venv` - Main App Runtime)
* **Target Mapping:** Android Runtime / ART Layer (Native Kotlin App).
* **Environment Optimization:** High-modernity dependencies (`numpy>=2.4.5`, `pydantic>=2.0`, `instructor`).
* **Functional Scope:** Orchestrates the master timeline control loops, evaluates semantic schema mapping properties, handles configuration routing, and hosts the Virtual Time Coordinate Transformer matrix.

### 2.2. Isolated Worker Sandbox (`.venv_ingest` - Native Subprocess Layer)
* **Target Mapping:** Android Native Development Kit (NDK) C++ Processing Space via JNI handles.
* **Environment Optimization:** Legacy-pinned, hardware-vector optimized binaries (`numpy<2`, PyTorch 2.2, Demucs Core, Faster-Whisper ASR).
* **Functional Scope:** Executes the heavy computational neural passes. It operates as an isolated command-line worker, ensuring heavy tensor libraries do not corrupt the parent process memory space.

---

## 3. Asymmetric Pipeline Partitioning Topology

```text
[ PRIMARY MASTER CONTEXT: .venv (NumPy 2.4) ]
   │
   ├──► 1. Environment Health Check (FFmpeg / Ollama Ping)
   ├──► 2. Launches Phase 1 via Explicit Path Redirection ──┐
   │                                                         │
   │  ┌──────────────────────────────────────────────────────┘
   │  ▼
   │ [ ISOLATED WORKER CONTEXT: .venv_ingest (NumPy 1.x) ]
   │    └──► Executes Demucs Audio Separation Pass
   │    └──► Executes Whisper ASR Segmentation Pass
   │    └──► Intercalates Dialogue Silence Gaps (Bug B4 Fix) ──► Dumps Checkpoint JSON
   │
   ├──► 3. Launches Phase 2 (Context-Aware Translation & Piper TTS)
   │       └──► Populates Running Timeline Clock Drift (Bug B11 Fix)
   │       └──► Compiles Virtual Seek Lookups (Action A12) ──► Dumps Master Manifest
   │
   └──► 4. Launches Phase 3: Bounded Sliding Viewport Playback Rendering (O(1) Memory Constant)

```

### 3.1. Phase 1: Asynchronous Static Ingestion Pass

* **Functional Tasks:**
* Computes an immutable file fingerprint using a 12-character SHA-256 slice.
* Isolates raw speech from room tone using Meta Demucs neural source separation.
* Extracts sub-millisecond word token boundaries using an optimized INT8 `whisper.cpp` engine combined with a Silero Voice Activity Filter (configured for a `600ms` minimum silence threshold).
* **Ambiance Intercalation Strategy:** Automatically detects conversational pauses between speech segments. If a gap exceeds $200\text{ms}$, it injects a synthetic `[SILENCE]` node into the manifest array. This ensures background room acoustics remain unbroken, preventing jump-cut skips downstream.



### 3.2. Phase 2: Synchronous Runtime Inference Queue

* **Functional Tasks:**
* Maintains a sliding lookahead processing window of exactly $N+2$ segments.
* **Sliding Window Context Lookahead:** When translating segment $N$, the system reads the text of segment $N+1$ and passes it into the LLM prompt. This lookahead context guides sentence structure and ensures natural grammatical transitions.
* Synthesizes speech via Piper TTS directly into volatile memory pipes, outputting unwarped local audio clips. It calibrates the conversational word targeting constant to $3.2$ words per second to match the native tempo of the Pratham engine.
* Compiles the final, event-driven data contract: `timeline_manifest.json`.



---

## 4. Bounded Sliding Viewport Playback Engine ($O(1)$ Memory Constant)

To prevent resource exhaustion on mobile platforms, the playback engine avoids building monolithic multi-read structures across an entire media timeline.

### 4.1. Memory Bounds Constraint

The execution footprint of the audio-video mixer is strictly bounded to a constant constraint ($O(1)$ Complexity). The system is prohibited from opening or tracking an arbitrary number of background audio streams ($N$) simultaneously:

$$\text{Active Stream Allocations} \le 3 \quad \forall \ N \in \text{Timeline}$$

Regardless of total video duration or segment density, the media engine maintains only three active tracking points: the active segment playing ($N$), the lookahead segment pre-buffered in memory ($N+1$), and the legacy segment undergoing garbage collection cleanups ($N-1$).

### 4.2. On-Demand Stream Mapping

Instead of splitting the master ambiance track up-front via massive `asplit` filter complexes, the system handles tracks dynamically. For each segment viewport, the player opens an independent, isolated media reader, binds it to the specific segment's timeline coordinates, runs the parametric ducking filters, and instantly drops the file descriptors the moment the playback head passes the segment boundary. This eliminates file-descriptor exhaustion bugs and keeps memory use stable.

---

## 5. Mathematical Foundations & Timeline State Machine

The pipeline maps the relationship between language expansion and video playback limits into three clean, non-destructive states based on millisecond tracking intervals.

### 5.1. Primary Routing Metric

The system evaluates the relationship between the natural untranslated speech duration ($D_{\text{nat}}$) and the original video segment window ($D_{\text{win}}$) to compute the speed ratio ($S_{\text{r}}$):

$$S_{\text{r}} = \frac{D_{\text{nat}}}{D_{\text{win}}}$$

To maintain natural conversational rhythm, the engine enforces a strict human pacing constraint:

$$0.85 \le S_{\text{r}} \le 1.20$$

### 5.2. State Routing Matrix

| Calculated Ratio | Assigned State | Vocal Track Execution Strategy | Video Track Playback Behavior |
| --- | --- | --- | --- |
| **$0.85 \le S_{\text{r}} \le 1.20$** | `NORMAL_SYNC` | Stream vocals at native $1.0\times$ human speed. | Regular $1:1$ forward frame progression. |
| **$S_{\text{r}} < 0.85$** | `PAD_EMPTY` | Stream vocals at native $1.0\times$ human speed. Pad trailing deficit with silence: <br>

<br> <br>$$\Delta D_{\text{pad}} = D_{\text{win}} - D_{\text{nat}}$$

 | Regular $1:1$ forward frame progression. |
| **$S_{\text{r}} > 1.20$** | `FREEZE_HOLD` | Stream vocals at native $1.0\times$ human speed. Zero pitch/cadence warping. | **Halt playback head.** Loop final frame image buffer for the overflow duration: <br>

<br> <br>$$\Delta D_{\text{freeze}} = D_{\text{nat}} - D_{\text{win}}$$

 |

---

## 6. Intercalated Virtual Timeline Coordinate Transformer Layout

Dynamic frameholds expand the total running duration of a lecture video. The system isolates the user interface from this expansion by implementing a bidirectional **Virtual Time Coordinate Transformer matrix**.

### 6.1. Timeline Coordination Mapping

The user interface interactions, scrubber layouts, and text caption timestamps bind exclusively to an expanded virtual timeline representation ($T_{\text{virtual}}$). This maps back to the physical video container timeline ($T_{\text{physical}}$) by tracking accumulated freeze-frame holds:

$$T_{\text{virtual}} = T_{\text{physical}} + \sum_{i=0}^{k} \Delta D_{\text{freeze}, i}$$

### 6.2. Scrubber Seek Interception Matrix

When a student moves the seek-bar to an explicit position ($T_{\text{virtual, target}}$), the transformer intercepts the event:

1. Performs a binary search lookup across the serialized manifest `virtual_timeline_map` array to instantly resolve the true target position on the read-only video track ($T_{\text{physical}}$).
2. Updates ExoPlayer's hardware decode head to that exact frame position.
3. Instantly flushes the volatile memory circular ring buffer.
4. Re-primes the $N+2$ lookahead inference queue at the new segment index to resume seamless playback under $180\text{ms}$.

---

## 7. Overlap-Add (OLA) Windowing Speech Synthesis Layout

To eliminate choppy transitions across segments, the system uses an Overlap-Add (OLA) Windowing Pipeline to preserve natural acoustic blending between sentences.

* **Context Extension Strategy:** The text parsing layer passes the first phrase of segment $N+1$ into the inference prompt for segment $N$ to build continuous phonetic context strings.
* **Crossfading Envelope:** At runtime, the trailing audio tail of segment $N$ and the leading edge of segment $N+1$ are mixed using a $80\text{ms}$ cross-fade. This transition is controlled by a continuous **Hann Window function envelope curve**:

$$w(n) = 0.5 \left(1 - \cos\left(\frac{2\pi n}{M-1}\right)\right), \quad 0 \le n \le M-1$$

This envelope smoothly blends the audio boundaries together, preserving a natural, continuous conversational rhythm.

---

## 8. Parametric Mid-Range Spectral Ducking DSP Specification

Lowering the overall volume of the background track during a freeze-frame drops the room tone completely, creating an unnatural "sonic vacuum" effect that tires the listener. The system resolves this using targeted **Spectral Ducking**.

* **Targeted Frequency Attenuation:** The background ambiance track (`no_vocals.wav`) is routed through an on-the-fly digital biquad IIR equalization filter block.
* **Vocal Formant Shielding:** When a Hindi vocal asset triggers, the filter attenuates only the mid-range speech frequency pocket spanning from **$1\text{kHz}$ to $3.5\text{kHz}$** by $-12\text{dB}$.
* **Acoustic Continuity Profile:** Low-frequency tones (projector fans, background hum) and high-frequency sounds remain untouched at $0\text{dB}$ attenuation. This keeps the background environment stable and eliminates the "sonic vacuum" effect.

```text
[ Ambiance Track Source ] ──► [ 1kHz - 3.5kHz Parametric Equalizer Filter ] ──┐
                                                  ▲                            ▼
                                    (Vocal Trigger Sidechain)            [ Audio Summer ] ──► Master Output
                                                  │                            ▲
[ Local Voice Stream Input ] ─────────────────────┴────────────────────────────┘

```

```

---

With your system blueprint updated to version 3.2.0, our data and platform contracts are locked in. 

Next, we will modify the phase controller scripts to execute this architecture cleanly. Let's begin by updating **`ingestion_pass.py`** to handle the path routing adjustments for the new `.venv_ingest` worker environment. Ready to update the ingestion script?

```