import os
import sys
import json
import wave
import struct
import subprocess
import urllib.request
from typing import Optional, Literal
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field, field_validator

import app_manifest

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
STUDIO_ASSETS = os.path.join(SCRIPT_ROOT, "studio_assets")
CHUNKS_DIR = os.path.join(STUDIO_ASSETS, "calibrated_chunks")
PIPER_MODELS_DIR = os.path.join(SCRIPT_ROOT, "piper_models")

class TranslationSchema(BaseModel):
    """Streamlined schema targeting optimized local token generation latency limits."""
    translated_text: str = Field(
        description="The natural academic Hindi/Hinglish translation written strictly in DEVANAGARI script. Do not include markdown or Latin characters."
    )
    # Resolves Finding N11: Restores strict validation guarantees on LLM structural planning
    syllable_strategy: Literal['expand', 'compress', 'normal'] = Field(default='normal')

    @field_validator('translated_text')
    @classmethod
    def verify_text_payload(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("The translated text field cannot be evaluated as empty.")
        return v

def bootstrap_piper_assets():
    """Resolves Finding N10: Stream downloads missing models with robust timeout protections."""
    os.makedirs(PIPER_MODELS_DIR, exist_ok=True)
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/hi/hi_IN/pratham/medium/"
    
    files = {
        "hi_IN-pratham-medium.onnx": base_url + "hi_IN-pratham-medium.onnx",
        "hi_IN-pratham-medium.onnx.json": base_url + "hi_IN-pratham-medium.onnx.json"
    }
    
    for filename, url in files.items():
        target_path = os.path.join(PIPER_MODELS_DIR, filename)
        if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
            print(f"   -> [Asset Bootstrapper] Downloading missing Piper voice asset: {filename}...")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as response:
                    with open(target_path, "wb") as out_file:
                        shutil_block = response.read(65536)
                        while shutil_block:
                            out_file.write(shutil_block)
                            shutil_block = response.read(65536)
            except Exception as e:
                raise RuntimeError(f"Failed to bootstrap critical model file {filename} from HuggingFace: {e}")

def generate_pure_silence_wav(output_path: str, duration_ms: int, sample_rate: int = 22050):
    """Generates a native 16-bit Mono PCM silence file to match Piper's audio profile."""
    num_frames = int((duration_ms / 1000.0) * sample_rate)
    silence_frame = struct.pack("<h", 0) # 16-bit mono zero frame (dropped down from stereo <hh)
    
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1) # ◄── FIXED: Changed from 2 to 1 to match native voice layouts
        wav_file.setsampwidth(2) # 16 bits
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(silence_frame * num_frames)

def query_local_translation_engine(english_text: str, target_duration_ms: int, next_context: Optional[str] = None) -> TranslationSchema:
    client = instructor.from_openai(
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"), mode=instructor.Mode.JSON
    )
    target_duration_sec = target_duration_ms / 1000.0
    computed_word_target = max(1, int(target_duration_sec * 3.2))
    context_str = f"Upcoming phrase context for lookahead blending: \"{next_context}\"" if next_context else "No upcoming lookahead context available."

    system_prompt = (
        "You are an expert academic translation engine operating within a synchronized video timeline framework.\n"
        "Your goal is to translate university computer science lectures into natural, clear Hinglish.\n\n"
        "CRITICAL SCRIPT RULE: You MUST write the final 'translated_text' value using DEVANAGARI SCRIPT (हिंदी लिपि) only. "
        "NEVER use Latin/English characters (A-Z) in the final translation string.\n\n"
        f"CRITICAL TIMING CONSTRAINT: The spoken translation must cleanly span approximately {target_duration_sec:.2f} seconds.\n"
        f"CONTEXT LOOKAHEAD: {context_str}"
    )
    
    try:
        response = client.chat.completions.create(
            model="gemma4:e2b",
            response_model=TranslationSchema,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Target Duration: {target_duration_sec:.2f}s | Target Word Count: {computed_word_target}w | Source Text: {english_text}"}
            ],
            temperature=0.2
        )
        return response
    except Exception as e:
        print(f"      ⚠️ Local translation node exception: {e}. Falling back to default baseline.")
        return TranslationSchema(translated_text="आगे बढ़ते हुए", syllable_strategy="normal")

def generate_unwarped_voice_piper(text: str, output_path: str) -> str:
    model_path = os.path.join(PIPER_MODELS_DIR, "hi_IN-pratham-medium.onnx")
    piper_bin = os.path.join(SCRIPT_ROOT, ".venv", "bin", "piper")
    if not os.path.exists(piper_bin):
        piper_bin = "piper"

    cmd = [piper_bin, "--model", model_path, "-f", output_path]
    try:
        subprocess.run(cmd, input=text, text=True, capture_output=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        # Resolves Finding N13: Fixed string attribute decoding exceptions during runtime panics
        err_msg = e.stderr.strip() if e.stderr else "Unknown Piper Crash Error"
        print(f"      ❌ Piper binary execution crash: {err_msg}")
        return None

def process_inference_queue(ingestion_payload_path: str) -> str:
    print(f"\n[PHASE 2] Initializing Synchronous Runtime Inference Queue pipeline...")
    bootstrap_piper_assets()

    with open(ingestion_payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    playback_timeline_nodes = []
    raw_segments = payload["raw_segments"]
    running_drift_accumulator = 0
    
    for i, seg in enumerate(raw_segments):
        idx = seg["segment_index"]
        is_gap = seg.get("is_gap", False)
        chunk_vocal_path = os.path.join(CHUNKS_DIR, f"vocal_chunk_{idx}.wav")

        if is_gap:
            print(f"\n   [Queue Node #{idx}] Processing Ambient Silence Gap ({seg['original_window_ms']}ms Allocation)")
            generate_pure_silence_wav(chunk_vocal_path, seg["original_window_ms"])
            directives = app_manifest.OrchestrationDirectives(
                timeline_action="NORMAL_SYNC", action_duration_ms=0, accumulated_drift_offset_ms=running_drift_accumulator
            )
            node = app_manifest.PlaybackSegmentNode(
                segment_index=idx, source_text="[SILENCE]", target_text="[SILENCE]",
                is_gap=True,
                anchor_timestamps=app_manifest.AnchorTimestamps(
                    start_ms=seg["start_ms"], end_ms=seg["end_ms"], original_window_ms=seg["original_window_ms"]
                ),
                audio_asset=app_manifest.AudioAssetMetadata(
                    local_path=os.path.relpath(chunk_vocal_path, SCRIPT_ROOT), natural_duration_ms=seg["original_window_ms"]
                ),
                orchestration=directives
            )
            playback_timeline_nodes.append(node)
            continue

        print(f"\n   [Queue Node #{idx}] Processing Lookahead Windows ({seg['original_window_ms']}ms Allocation)")
        print(f"      -> Source English: \"{seg['text']}\"")
        
        # Resolves Finding N12: Skip empty gap items to pass meaningful downstream lookahead parameters
        next_context = None
        for lookahead_ptr in range(i + 1, len(raw_segments)):
            if not raw_segments[lookahead_ptr].get("is_gap", False):
                next_context = raw_segments[lookahead_ptr]["text"]
                break

        llm_payload = query_local_translation_engine(seg["text"], seg["original_window_ms"], next_context)
        print(f"      -> Target Devanagari: \"{llm_payload.translated_text}\"")
        
        generated_asset_path = generate_unwarped_voice_piper(llm_payload.translated_text, chunk_vocal_path)
        
        if not generated_asset_path or os.path.getsize(generated_asset_path) == 0:
            generate_pure_silence_wav(chunk_vocal_path, seg["original_window_ms"])
            natural_duration_ms = seg["original_window_ms"]
            llm_payload.translated_text = "[FALLBACK]"
        else:
            cmd_probe = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", generated_asset_path
            ]
            try:
                natural_duration_ms = int(float(subprocess.check_output(cmd_probe).strip()) * 1000)
            except Exception:
                natural_duration_ms = seg["original_window_ms"]

        # Resolves Finding N8: Inject cumulative timeline coordinates at construction time
        directives = app_manifest.compute_orchestration_directives(
            seg["original_window_ms"], natural_duration_ms, running_drift_accumulator
        )
        running_drift_accumulator += directives.action_duration_ms if directives.timeline_action == "FREEZE_HOLD" else 0
        print(f"      -> Assigned State Directive: {directives.timeline_action} (Delta: {directives.action_duration_ms}ms)")

        node = app_manifest.PlaybackSegmentNode(
            segment_index=idx, source_text=seg["text"], target_text=llm_payload.translated_text,
            is_gap=False,
            anchor_timestamps=app_manifest.AnchorTimestamps(
                start_ms=seg["start_ms"], end_ms=seg["end_ms"], original_window_ms=seg["original_window_ms"]
            ),
            audio_asset=app_manifest.AudioAssetMetadata(
                local_path=os.path.relpath(chunk_vocal_path, SCRIPT_ROOT), natural_duration_ms=natural_duration_ms
            ),
            orchestration=directives
        )
        playback_timeline_nodes.append(node)

    master_manifest = app_manifest.LectureDubbingManifest(
        source_video_hash=payload["source_video_hash"], total_duration_seconds=payload["total_duration_seconds"],
        total_segments=len(playback_timeline_nodes), playback_timeline=playback_timeline_nodes
    )
    
    output_manifest_json = os.path.join(STUDIO_ASSETS, f"timeline_manifest_{payload['source_video_hash']}.json")
    master_manifest.serialize_manifest(output_manifest_json)
    return output_manifest_json