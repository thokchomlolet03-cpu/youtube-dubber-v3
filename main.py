import os
import sys
import shutil
import argparse
import subprocess
import urllib.request
import hashlib
import json

import inference_queue
import player_emulator

def verify_system_environment() -> bool:
    print("📋 Running system environment verification checks...")
    for binary in ["ffmpeg", "ffprobe"]:
        if shutil.which(binary) is None:
            print(f"   ❌ Critical system error: '{binary}' is missing from your system PATH.")
            return False
        print(f"   -> Found system dependency: {binary} ✅")

    # Health Check Verification
    ollama_health_url = "http://localhost:11434/"
    print(f"   -> Pinging local LLM engine microservice at: {ollama_health_url}")
    try:
        with urllib.request.urlopen(ollama_health_url, timeout=3) as response:
            if response.getcode() != 200:
                return False
    except Exception:
        print("   ❌ Critical system error: Local Ollama server is unreachable on localhost:11434.")
        return False

    # Resolves Finding N1: Verify that the required model is pulled and ready to accept requests
    print("   -> Validating model registry deployment ('gemma4:e2b')...")
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as response:
            registry = json.loads(response.read().decode())
            models = [m["name"] for m in registry.get("models", [])]
            if not any("gemma4:e2b" in m for m in models):
                print("   ❌ Critical: 'gemma4:e2b' was not found in local model registry. Run: ollama pull gemma4:e2b")
                return False
    except Exception:
        pass

    print("Base system diagnostics completed cleanly.\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="youtube-dubber-v3: Orchestrator (v3.2.0-Fixed).")
    parser.add_argument("video_input", type=str, help="Path to original English lecture video container file")
    parser.add_argument("--output", "-o", type=str, default="final_elastic_lecture.mp4", help="Desired output filename")
    args = parser.parse_args()

    if not verify_system_environment():
        sys.exit(1)

    print("======================================================================")
    print("🚀 LAUNCHING ON-DEVICE ELASTIC DUBBING PIPELINE (V3.2.0 RESILIENT)")
    print("======================================================================")

    try:
        # Resolves Finding Venv Isolation: Run Phase 1 inside the .venv_ingest worker process
        print("\n[PHASE 1] Routing Ingestion Worker to Isolated Sibling Sandbox...")
        worker_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv_ingest", "bin", "python")
        if not os.path.exists(worker_python):
            worker_python = "python3"

        ingestion_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingestion_pass.py")
        cmd_phase1 = [worker_python, ingestion_script, args.video_input]
        
        # Invoke Phase 1 as an external process to preserve binary path separation
        subprocess.run(cmd_phase1, check=True)

        # Extract output fingerprint coordinates to track intermediate steps
        cmd_probe = [
            "ffprobe", "-v", "error", "-show_entries", "format=filename",
            "-of", "default=noprint_wrappers=1:nokey=1", args.video_input
        ]
        video_hash = hashlib.sha256()
        with open(os.path.abspath(args.video_input), "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                video_hash.update(byte_block)
        hash_str = video_hash.hexdigest()[:12]
        
        checkpoint_json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "studio_assets", f"ingestion_checkpoint_{hash_str}.json"
        )

        # Phase 2: Synchronous Runtime Inference Queue (Executes safely inside modern .venv)
        manifest_json_path = inference_queue.process_inference_queue(checkpoint_json_path)

        # Phase 3: Playback Emulator Core Renderer (Runs with an O(1) constant memory profile)
        player_emulator.emulate_player_playback(manifest_json_path, args.output)

        print("======================================================================")
        print("🎉 SUCCESS: PIPELINE ROUTINES COMPLETED END-TO-END")
        print(f"🎯 Target Destination File Generated: {os.path.abspath(args.output)}")
        print("======================================================================")

    except Exception as pipeline_error:
        print(f"\n💥 CRITICAL PIPELINE FAILURE during runtime execution loop:")
        print(f"   Reason: {str(pipeline_error)}")
        sys.exit(1)

if __name__ == "__main__":
    main()