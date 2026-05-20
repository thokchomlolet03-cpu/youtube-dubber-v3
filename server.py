import os
import shutil
import uuid
import hashlib
import json
import threading
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

import inference_queue
import player_emulator

app = FastAPI(
    title="OmniLect Local Server Engine",
    description="M1 Mac Local loopback processing engine for automated lecture translation"
)

# Workspace directories inside your project folder
WORKSPACE_DIR = Path("./omnilect_workspace")
OUTPUT_ZIP_DIR = Path("./compiled_packages")

WORKSPACE_DIR.mkdir(exist_ok=True)
OUTPUT_ZIP_DIR.mkdir(exist_ok=True)

# Global lock to serialize pipeline executions and prevent CPU overload
pipeline_lock = threading.Lock()

def run_pipeline_real(video_id: str, video_path: Path, scratch_path: Path):
    """
    Executes the real processing pipeline:
      1. Computes SHA-256 hash of the video.
      2. Runs Phase 1 (Ingestion) as an isolated subprocess using the .venv_ingest Python binary.
      3. Runs Phase 2 (Inference Queue) to perform Ollama translation and Piper TTS synthesis.
      4. Generates the Android Media3 composition JSON.
      5. Copies the background track (no_vocals.wav) and all vocal chunks to the scratch folder.
      6. Writes the final manifest.json.
    """
    video_abs_path = str(video_path.resolve())
    
    # 1. Compute video hash
    sha256_hash = hashlib.sha256()
    with open(video_abs_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    video_hash = sha256_hash.hexdigest()[:12]
    
    # 2. Run Phase 1 Ingestion as an isolated worker subprocess
    script_dir = Path(__file__).parent.resolve()
    worker_python = script_dir / ".venv_ingest" / "bin" / "python"
    if not worker_python.exists():
        worker_python = Path("python3")
        
    ingestion_script = script_dir / "ingestion_pass.py"
    
    print(f"[{video_id}] Starting ingestion subprocess...")
    cmd_phase1 = [str(worker_python), str(ingestion_script), video_abs_path]
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]
        
    try:
        subprocess.run(cmd_phase1, env=env, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ingestion worker failed with exit code: {e.returncode}")
        
    checkpoint_json_path = script_dir / "studio_assets" / f"ingestion_checkpoint_{video_hash}.json"
    if not checkpoint_json_path.exists():
        raise FileNotFoundError(f"Ingestion checkpoint file not found at: {checkpoint_json_path}")
        
    # 3. Run Phase 2 (Inference Queue) to generate the manifest & audio chunks
    print(f"[{video_id}] Running inference queue...")
    manifest_json_path = inference_queue.process_inference_queue(str(checkpoint_json_path))
    
    # Load manifest and checkpoint data
    with open(manifest_json_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    with open(checkpoint_json_path, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)
        
    no_vocals_wav_path = Path(checkpoint_data["no_vocals_wav_path"])
    if not no_vocals_wav_path.is_absolute():
        no_vocals_wav_path = script_dir / no_vocals_wav_path
        
    # 4. Generate the Media3 Composition Payload
    mobile_payload = player_emulator.generate_mobile_media3_composition(manifest_data, str(no_vocals_wav_path))
    
    # 5. Populate the scratch workspace for packaging
    scratch_path.mkdir(parents=True, exist_ok=True)
    
    # Mux original video stream and demuxed ambient audio stream into a single MP4 file
    no_vocals_mp4_path = scratch_path / "no_vocals.mp4"
    print(f"[{video_id}] Muxing video {video_path} and ambient audio {no_vocals_wav_path} into {no_vocals_mp4_path}...")
    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path.resolve()),
            "-i", str(no_vocals_wav_path.resolve()),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(no_vocals_mp4_path)
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[-] FFmpeg muxing failed: {e}. Falling back to copying original video.")
        shutil.copy2(video_path, no_vocals_mp4_path)
        
    mobile_payload["base_background_audio_track"] = "no_vocals.mp4"
    
    # Copy vocal chunks and update their paths in composition playlist
    for segment in mobile_payload["composition_playlist"]:
        vocal_local_path = segment["vocal_asset_local_path"]
        vocal_abs_path = script_dir / vocal_local_path
        
        vocal_filename = vocal_abs_path.name
        shutil.copy2(vocal_abs_path, scratch_path / vocal_filename)
        
        # Flatten the path in the composition playlist manifest to be relative to the zip root
        segment["vocal_asset_local_path"] = vocal_filename
        
    # Write the modified composition JSON to the scratch folder
    with open(scratch_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(mobile_payload, f, ensure_ascii=False, indent=2)

@app.get("/translate/{video_id}")
def process_video_translation(video_id: str):
    """
    Endpoint that the Android app calls via 10.0.2.2:8000
    """
    # 1. First level short-circuit check: check by video_id name
    cached_zip_by_id = OUTPUT_ZIP_DIR / f"pack_{video_id}.zip"
    if cached_zip_by_id.exists():
        print(f"⚡ [CACHE HIT] Found pre-compiled package for {video_id}. Serving instantly.")
        return FileResponse(
            path=cached_zip_by_id, 
            media_type="application/zip", 
            filename=f"pack_{video_id}.zip"
        )

    # 2. Resolve video_id to a local video file (search in root project folder)
    script_dir = Path(__file__).parent.resolve()
    video_path = script_dir / f"{video_id}.mp4"
    
    video_hash = None
    # Check if video_id is a fingerprint by looking up checkpoint files first
    if not video_path.exists():
        checkpoint_file = script_dir / "studio_assets" / f"ingestion_checkpoint_{video_id}.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                resolved_path = Path(checkpoint_data.get("source_video_path", ""))
                if resolved_path.exists():
                    video_path = resolved_path
                    video_hash = video_id
            except Exception as e:
                print(f"Error reading checkpoint for fingerprint lookup: {e}")
                
    # Fallback to scanning all .mp4 files in the workspace and computing their hashes
    if not video_path.exists():
        for file in script_dir.glob("*.mp4"):
            sha256_hash = hashlib.sha256()
            try:
                with open(file, "rb") as f:
                    for byte_block in iter(lambda: f.read(65536), b""):
                        sha256_hash.update(byte_block)
                file_hash = sha256_hash.hexdigest()[:12]
                if file_hash == video_id:
                    video_path = file
                    video_hash = video_id
                    break
            except Exception as e:
                print(f"Error computing hash for {file}: {e}")

    # Fallback to direct path check if it's already an absolute or relative path that exists
    if not video_path.exists():
        video_path = Path(video_id)
        if not video_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Video file not found for ID: {video_id}. Make sure {video_id}.mp4 exists in the workspace root."
            )

    # Resolve video_hash if not already known
    if not video_hash:
        # Check if we can get it from a cached checkpoint first to avoid reading the whole file
        for cp_file in (script_dir / "studio_assets").glob("ingestion_checkpoint_*.json"):
            try:
                with open(cp_file, "r", encoding="utf-8") as f:
                    cp_data = json.load(f)
                if cp_data.get("source_video_path") == str(video_path.resolve()):
                    video_hash = cp_data.get("source_video_hash")
                    break
            except Exception:
                pass
        
        # Fallback to computing hash if not found in checkpoints
        if not video_hash:
            sha256_hash = hashlib.sha256()
            with open(video_path, "rb") as f:
                for byte_block in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(byte_block)
            video_hash = sha256_hash.hexdigest()[:12]

    # 3. Second level short-circuit check: check by resolved video_hash
    if video_hash and video_hash != video_id:
        cached_zip_by_hash = OUTPUT_ZIP_DIR / f"pack_{video_hash}.zip"
        if cached_zip_by_hash.exists():
            print(f"⚡ [CACHE HIT] Found pre-compiled package for hash {video_hash}. Serving instantly.")
            return FileResponse(
                path=cached_zip_by_hash, 
                media_type="application/zip", 
                filename=f"pack_{video_hash}.zip"
            )

    session_id = f"job_{video_id}_{uuid.uuid4().hex[:6]}"
    scratch_path = WORKSPACE_DIR / session_id
    
    # 4. Acquire lock and run the real pipeline
    print(f"Received request to translate: {video_id} (Resolved hash: {video_hash})")
    acquired = pipeline_lock.acquire(timeout=600.0) # 10 minutes timeout if queue is deep
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="Server is busy processing other video translation requests. Please try again later."
        )
        
    try:
        run_pipeline_real(video_id, video_path, scratch_path)
        
        # 5. Zip everything up into a clean package for the phone (using the unique video_hash for the filename)
        archive_target = OUTPUT_ZIP_DIR / f"pack_{video_hash}"
        zip_file_path = shutil.make_archive(
            base_name=str(archive_target), 
            format="zip", 
            root_dir=str(scratch_path)
        )
        
        return FileResponse(
            path=zip_file_path, 
            media_type="application/zip", 
            filename=f"pack_{video_hash}.zip"
        )
        
    except Exception as e:
        print(f"Error during video translation: {e}")
        raise HTTPException(status_code=500, detail=f"Server Pipeline Error: {str(e)}")
        
    finally:
        # 6. Clean up scratch directory and release lock
        if scratch_path.exists():
            shutil.rmtree(scratch_path)
        pipeline_lock.release()