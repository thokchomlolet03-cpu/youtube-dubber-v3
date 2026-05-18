Technical Specification & Architecture Blueprint v3.1: On-Device Elastic Dubbing Engine
Document Reference: SPEC-EDB-V31
Target Goal: 100% Offline, On-Device, High-Performance Academic Video Translation for Global Educational Accessibility

1. Executive Summary & System Goals
The v3.1 architecture bridges the gap between desktop AI prototypes and mobile production environments. Designed for mobile hardware, this system processes instructional videos entirely on-device, bypassing cloud API dependencies. By shifting from destructive digital signal processing (DSP) time-stretching to a Manifest-Driven Elastic Timeline, the framework ensures natural human speech delivery while keeping video frames accurately locked to the educational materials.

2. Asymmetric Pipeline Partitioning (Computing Topology)
To prevent thermal throttling and ensure a smooth user interface, the system splits processing into an upfront Asynchronous Static Ingestion Pass and a real-time Synchronous Runtime Inference Queue.
[ PHASE 1: ASYNCHRONOUS STATIC INGESTION ]
  └── Ingest Video File ──► Demucs (Vocal Split) ──► whisper.cpp (INT8 ASR) ──► Generate Base Structural Map

[ PHASE 2: SYNCHRONOUS RUNTIME INFERENCE ]
  └── Playback Head ──► Maintain N+2 Lookahead Window ──► On-Device LLM ──► Piper TTS (Raw PCM Streams)


2.1. Asynchronous Static Ingestion Pass
Execution Trigger: Initial file import or download.
Compute Footprint: High-utilization, pinned to performance cores (ARM Big/Prime).
Operations:
Extract structural audio stems via Meta Demucs ($htdemucs$ convolution layers).
Compute token timestamps via whisper.cpp using an INT8 quantized model.
Cache the extracted background ambiance stem (no_vocals.wav) and the raw English text matrix to disk.
2.2. Synchronous Runtime Inference Queue
Execution Trigger: Active video playback loop.
Compute Footprint: Ultra-low overhead, offloaded to the device's hardware NPU (Neural Processing Unit) via Android's NNAPI or Apple's CoreML.
Operations:
Maintains a strict $N+2$ lookahead segment window.
Translates text to Devanagari Hinglish using an INT4 local LLM.
Synthesizes speech via the Piper TTS engine directly into memory.

3. Mathematical Foundations & Timeline State Machine
The timeline engine treats speech pacing as a variable function of linguistic length against video container constraints, avoiding artificial audio warping.
3.1. Core Metrics & Constraints
The system measures the relationship between the natural untranslated speech duration ($D_{\text{nat}}$) and the original video segment window ($D_{\text{win}}$) to compute the speed ratio ($S_{\text{r}}$):
$$S_{\text{r}} = \frac{D_{\text{nat}}}{D_{\text{win}}}$$
To maintain natural human vocal cadence, the system enforces a strict pacing threshold constraint:
$$0.85 \le S_{\text{r}} \le 1.20$$
3.2. State Machine Routing Matrix
                         ┌───────────────────────────┐
                          │ Calculate Speed Ratio     │
                          │   Sr = D_nat / D_win      │
                          └─────────────┬─────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
           [ Sr < 0.85 ]          [ 0.85 <= Sr <= 1.20 ]     [ Sr > 1.20 ]
                 │                      │                      │
                 ▼                      ▼                      ▼
          (State: PAD_EMPTY)     (State: NORMAL_SYNC)   (State: FREEZE_HOLD)
          Pad with silence       Standard 1:1 playback  Freeze video frame


STATE: NORMAL_SYNC ($0.85 \le S_{\text{r}} \le 1.20$)
The synthesized vocal stream plays at native $1.0\times$ speed. Video tracks run at a standard $1:1$ forward frame rate.
STATE: PAD_EMPTY ($S_{\text{r}} < 0.85$)
The vocal track plays at native $1.0\times$ speed. The remaining timeline deficit is automatically padded with digital silence:
$$\Delta D_{\text{pad}} = D_{\text{win}} - D_{\text{nat}}$$
STATE: FREEZE_HOLD ($S_{\text{r}} > 1.20$)
The vocal track plays at a clean $1.0\times$ human pace. The video playback engine halts at the segment's boundary marker, looping the final frame image buffer for the exact duration of the calculated overflow:
$$\Delta D_{\text{freeze}} = D_{\text{nat}} - D_{\text{win}}$$

4. In-Memory Volatile Streaming Interface Specification
To prevent write amplification jank and protect device flash memory lifespans, the runtime engine replaces disk storage with a lock-free In-Memory Volatile Circular Ring Buffer.
Memory Allocation Architecture: The application claims a volatile, anonymous memory block (ashmem or native AHardwareBuffer via JNI pointers) capped at a fixed storage allocation:
$$\text{Buffer Size} = \text{Sample Rate} \times \text{Bit Depth} \times \text{Channels} \times \text{Window Lookahead}$$
For 44100Hz 16-bit stereo across a 3-segment window, this holds a static RAM footprint under $15\text{MB}$.
Producer/Consumer Contract: The Piper TTS engine acts as a stream producer, piping raw PCM byte blocks directly to the ring buffer's head pointer. ExoPlayer acts as the consumer, reading data directly from the trailing buffer tail pointer, completely avoiding disk I/O operations.

5. Intercalated Virtual Timeline Coordinate Transformer Layout
Dynamic video freezing shifts the relationship between the absolute video track timeline and the actual lesson experience. The system handles this by introducing a Virtual Time Coordinate Transformer matrix.
5.1. Mathematical Timeline Mapping
The student interface interacts exclusively with a virtual timeline representation ($T_{\text{virtual}}$), which maps back to the physical video track position ($T_{\text{physical}}$) by factoring in cumulative freeze deltas:
$$T_{\text{virtual}} = T_{\text{physical}} + \sum_{i=0}^{k} \Delta D_{\text{freeze}, i}$$
5.2. Scrubber Seek Interception Matrix
When a student interacts with the seek bar at $T_{\text{virtual, target}}$, the player runs a binary search lookup to resolve the true video frame target:
Plaintext
[ Scrubber Touch: T_virtual ] 
            │
            ▼ (Binary Search Lookup Table)
[ Evaluate Cumulative Freezes ] ──► Compute T_physical ──► Seek ExoPlayer Head
            │
            ▼
[ Flush Ring Buffer ] ──► Re-prime Lookahead Queue at New Segment Index



6. Overlap-Add (OLA) Windowing Speech Synthesis Layout
To eliminate choppy transitions across segments, the system uses an Overlap-Add (OLA) Windowing Pipeline to create smooth phonetic boundaries.
Context Extension Strategy: The text parsing layer passes the first phrase of segment $N+1$ into the inference prompt for segment $N$. This builds a continuous phonetic transition string.
Crossfading Envelope: At runtime, the trailing audio tail of segment $N$ and the leading edge of segment $N+1$ are mixed using a $50\text{ms}$ to $100\text{ms}$ cross-fade. This transition is controlled by a continuous Hann Window function envelope curve:
$$w(n) = 0.5 \left(1 - \cos\left(\frac{2\pi n}{M-1}\right)\right), \quad 0 \le n \le M-1$$
This envelope smoothly blends the audio boundaries together, preserving a natural, continuous conversational rhythm.

7. Parametric Mid-Range Spectral Ducking DSP Specification
To ensure high speech intelligibility while protecting natural room acoustics, the system replaces wide-band volume drops with targeted Spectral Ducking.
Targeted Frequency Attenuation: The background ambiance track (no_vocals.wav) is routed through an on-the-fly digital biquad IIR equalization filter block.
Vocal Formant Shielding: When a Hindi vocal asset triggers, the filter attenuates only the mid-range speech frequency pocket spanning from $1\text{kHz}$ to $3.5\text{kHz}$ by $-12\text{dB}$.
Acoustic Continuity Profile: Low-frequency tones (projector fans, background hum) and high-frequency sounds remain untouched at $0\text{dB}$ attenuation. This keeps the background environment stable and eliminates the "sonic vacuum" effect.
                 DIGITAL COMPONENT MIXING ROUTE
                  
[ Background Track ] ──► [ 1kHz - 3.5kHz Parametric Filter ] ──┐
                                 ▲                            ▼
                       (Vocal Detection Sidechain)      [ Audio Summer ] ──► Master Output
                                 │                            ▲
[ Local Voice Stream ] ──────────┴────────────────────────────┘



8. Android Native Migration Strategy (Kotlin/ExoPlayer Mapping)
When translating this validated Python mock prototype into native Android Kotlin source code, the operational architecture maps directly to these system components:
Audio Engine Integration: Reallocate Python's raw PCM buffer writes over to a native Android AudioTrack structure initialized within a low-latency streaming loop.
Custom Media Source: Subclass ExoPlayer's Core.MediaSource layer to build a composite playback framework capable of handling simultaneous video rendering and custom memory-mapped audio stream ingestion.
Dynamic Timeline Interception: Bind the virtual timeline mapping transformer directly into ExoPlayer's Player.Listener.onPositionDiscontinuity interface. This allows the system to handle frame-freezing loops programmatically without breaking native player behaviors.
Hardware Acceleration Interface: Port the local translation and TTS model tasks to run via ONNX Runtime Mobile, binding model instructions directly to the device's hardware NPU through Android's Neural Networks API (NNAPI).

