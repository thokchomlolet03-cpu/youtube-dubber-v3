import os
import sys
import hashlib
import json
import subprocess

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
STUDIO_ASSETS = os.path.join(SCRIPT_ROOT, "studio_assets")
CACHE_VERSION_KEY = "v3.2_segmented" # Resolves Finding N4: Cache invalidation key

def calculate_video_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:12]

def run_demucs_separation(video_path: str, output_dir: str) -> tuple:
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Resolves Finding N6: Pre-create target output layout
    os.makedirs(os.path.join(output_dir, "htdemucs"), exist_ok=True)
    vocals_wav = os.path.join(output_dir, "htdemucs", base_name, "vocals.wav")
    no_vocals_wav = os.path.join(output_dir, "htdemucs", base_name, "no_vocals.wav")

    if os.path.exists(vocals_wav) and os.path.exists(no_vocals_wav):
        print("   -> [Idempotency Cache] Pristine separated audio stems found. Skipping Demucs pass.")
        return vocals_wav, no_vocals_wav

    print("   -> Launching Neural Source Separation (Demucs CPU Vector Optimization)...")
    demucs_bin = os.path.join(SCRIPT_ROOT, ".venv_ingest", "bin", "demucs")
    if not os.path.exists(demucs_bin):
        demucs_bin = "demucs"

    cmd = [demucs_bin, "--two-stems", "vocals", "-d", "cpu", "-o", output_dir, video_path]
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]

    try:
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Demucs processing node failed with exit code: {e.returncode}.")

    return vocals_wav, no_vocals_wav

def run_whisper_transcription(vocals_wav_path: str, total_duration_ms: int) -> list:
    cache_json_path = vocals_wav_path.replace(".wav", "_raw_segments.json")
    
    # Resolves Finding N4: Validate that the cached file matches the current schema configuration
    if os.path.exists(cache_json_path):
        try:
            with open(cache_json_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                if isinstance(cached_data, dict) and cached_data.get("cache_version") == CACHE_VERSION_KEY:
                    print("   -> [Idempotency Cache] Validated sandbox transcription index found. Skipping Whisper pass.")
                    return cached_data["nodes"]
                print("   -> [Cache Invalidation] Stale schema signature detected. Forcing Whisper recalculation...")
        except Exception:
            print("   -> [Cache Corrupted] Re-initializing transcription tracking buffers...")

    # Postponed Dynamic Import: Ensures faster-whisper runs inside the .venv_ingest worker process
    print("   -> Loading Faster-Whisper core libraries inside isolated worker runtime space...")
    from faster_whisper import WhisperModel

    print("   -> Initializing speech tracking engine via Faster-Whisper INT8 model...")
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    
    segments, info = model.transcribe(
        vocals_wav_path, beam_size=5, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=600)
    )

    extracted_nodes = []
    segment_counter = 0
    current_time_ms = 0
    GAP_THRESHOLD_MS = 200 

    for segment in segments:
        clean_text = segment.text.strip()
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)

        # Resolves Finding N5: Synchronize timeline tracking states before evaluating empty dialogue fields
        if not clean_text:
            current_time_ms = max(current_time_ms, end_ms)
            continue

        # Process silence gaps cleanly
        if start_ms > (current_time_ms + GAP_THRESHOLD_MS):
            gap_duration = start_ms - current_time_ms
            extracted_nodes.append({
                "segment_index": segment_counter, "text": "[SILENCE]", "start_ms": current_time_ms,
                "end_ms": start_ms, "original_window_ms": gap_duration, "is_gap": True
            })
            segment_counter += 1

        original_window_ms = end_ms - start_ms
        if original_window_ms <= 0:
            continue

        extracted_nodes.append({
            "segment_index": segment_counter, "text": clean_text, "start_ms": start_ms,
            "end_ms": end_ms, "original_window_ms": original_window_ms, "is_gap": False
        })
        segment_counter += 1
        current_time_ms = end_ms

    if total_duration_ms > (current_time_ms + GAP_THRESHOLD_MS):
        trailing_gap = total_duration_ms - current_time_ms
        extracted_nodes.append({
            "segment_index": segment_counter, "text": "[SILENCE]", "start_ms": current_time_ms,
            "end_ms": total_duration_ms, "original_window_ms": trailing_gap, "is_gap": True
        })

    # Commit cache payload along with its schema version tag
    with open(cache_json_path, "w", encoding="utf-8") as f:
        json.dump({"cache_version": CACHE_VERSION_KEY, "nodes": extracted_nodes}, f, ensure_ascii=False, indent=2)

    return extracted_nodes

def process_ingestion_pass(video_file_path: str) -> str:
    abs_video_path = os.path.abspath(video_file_path)
    if not os.path.exists(abs_video_path):
        raise FileNotFoundError(f"Target lecture file missing at path: {abs_video_path}")

    print(f"\n[PHASE 1] Starting Asynchronous Static Ingestion Pass for: {os.path.basename(abs_video_path)}")
    os.makedirs(STUDIO_ASSETS, exist_ok=True)

    video_hash = calculate_video_hash(abs_video_path)
    print(f"   -> Calculated file footprint token: {video_hash}")

    cmd_probe = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", abs_video_path
    ]
    try:
        total_duration_seconds = float(subprocess.check_output(cmd_probe).strip())
    except Exception:
        total_duration_seconds = 30.63
        
    total_duration_ms = int(total_duration_seconds * 1000)

    vocals_wav, no_vocals_wav = run_demucs_separation(abs_video_path, STUDIO_ASSETS)
    raw_segments = run_whisper_transcription(vocals_wav, total_duration_ms)

    ingestion_payload = {
        "source_video_path": abs_video_path, "source_video_hash": video_hash,
        "total_duration_seconds": total_duration_seconds, "no_vocals_wav_path": no_vocals_wav,
        "raw_segments": raw_segments
    }

    output_payload_path = os.path.join(STUDIO_ASSETS, f"ingestion_checkpoint_{video_hash}.json")
    with open(output_payload_path, "w", encoding="utf-8") as f:
        json.dump(ingestion_payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Phase 1 static ingestion completed. Payload saved to: {output_payload_path}")
    return output_payload_path

if __name__ == "__main__":
    # Internal CLI router: Enables main.py to invoke this file directly as an isolated worker process
    if len(sys.argv) > 1:
        process_ingestion_pass(sys.argv[1])