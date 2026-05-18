import os
import sys
import json
import subprocess
from typing import Literal
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field, field_validator

# Import our Pydantic manifest contract engine from the local sandbox directory
import app_manifest

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
STUDIO_ASSETS = os.path.join(SCRIPT_ROOT, "studio_assets")
CHUNKS_DIR = os.path.join(STUDIO_ASSETS, "calibrated_chunks")

class TranslationSchema(BaseModel):
    """Streamlined schema targeting optimized local token generation latency limits."""
    translated_text: str = Field(
        description="The natural academic Hindi/Hinglish translation written strictly in DEVANAGARI script. Do not include markdown or Latin characters."
    )
    syllable_strategy: Literal['expand', 'compress', 'normal'] = Field(
        description="Pacing strategy required to fit the temporal budget constraints."
    )

    @field_validator('translated_text')
    @classmethod
    def verify_text_payload(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("The translated text field cannot be evaluated as empty.")
        return v

def query_local_translation_engine(english_text: str, target_duration_ms: int) -> TranslationSchema:
    """Leverages local LLM quantization layers via Instructor to fetch Devanagari mappings."""
    client = instructor.from_openai(
        OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
        mode=instructor.Mode.JSON
    )
    
    target_duration_sec = target_duration_ms / 1000.0
    computed_word_target = max(1, int(target_duration_sec * 2.8))
    
    system_prompt = (
        "You are an expert academic translation engine operating within a synchronized video timeline framework.\n"
        "Your goal is to translate university computer science lectures into natural, clear Hinglish.\n\n"
        "CRITICAL SCRIPT RULE: You MUST write the final 'translated_text' value using DEVANAGARI SCRIPT (हिंदी लिपि) only. "
        "NEVER use Latin/English characters (A-Z) in the final translation string. "
        "Transliterate English technical nouns directly into phonetic Devanagari characters. "
        "For example, write 'professor' as 'प्रोफेसर', 'tool' as 'टूल', 'communication' as 'कम्युनिकेशन', and 'bright' as 'ब्राइट'.\n\n"
        f"CRITICAL TIMING CONSTRAINT: The spoken translation must cleanly span approximately {target_duration_sec:.2f} seconds.\n"
        f"To achieve a natural conversational pace, your translation should hit a target size of roughly {computed_word_target} words.\n"
        "- If the target window is long relative to the text, EXPAND the phrasing naturally using descriptive vocabulary.\n"
        "- If the target window is tight, COMPRESS the phrasing down using concise expressions.\n\n"
        "CRITICAL SYNTAX RULE: Always preserve the speaker's core active agency. Ensure all verb agreements match pluralities."
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
        # Secure fallback mapping directly to Devanagari to protect phonemizer layers
        return TranslationSchema(translated_text="नमस्ते", syllable_strategy="normal")

def generate_unwarped_voice_piper(text: str, output_path: str) -> str:
    """Synthesizes raw 16-bit vocal layers locked permanently at a native 1.0x human pace."""
    piper_bin = os.path.join(SCRIPT_ROOT, ".venv", "bin", "piper")
    model_path = os.path.join(SCRIPT_ROOT, "piper_models", "hi_IN-pratham-medium.onnx")
    
    if not os.path.exists(piper_bin):
        piper_bin = "piper" # Fall back to PATH configuration bounds if local venv binary is elusive

    cmd = [
        piper_bin,
        "--model", model_path,
        "--output_file", output_path
    ]
    try:
        subprocess.run(cmd, input=text, text=True, capture_output=True, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"      ❌ Piper binary execution crash: {e.stderr}")
        return None

def extract_audio_duration_ms(audio_path: str) -> int:
    """Reads the exact physical duration of the generated TTS file in milliseconds."""
    # REVISED BUG #2 FIX: Corrected ffprobe option mapping using nokey=1 parameters
    cmd_probe = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    try:
        duration_seconds = float(subprocess.check_output(cmd_probe).strip())
        return int(duration_seconds * 1000)
    except Exception:
        # Secure fallback calculation if ffprobe encounters read tracking locks
        return 0

def process_inference_queue(ingestion_payload_path: str) -> str:
    """Orchestrates Phase 2 processing loops to compile the master timeline JSON contract."""
    print(f"\n[PHASE 2] Initializing Synchronous Runtime Inference Queue pipeline...")
    
    with open(ingestion_payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    os.makedirs(CHUNKS_DIR, exist_ok=True)
    
    playback_timeline_nodes = []
    
    # Process each raw segment extracted during the initial ingestion pass
    for seg in payload["raw_segments"]:
        idx = seg["segment_index"]
        print(f"\n   [Queue Node #{idx}] Processing Lookahead Windows ({seg['original_window_ms']}ms Allocation)")
        print(f"      -> Source English: \"{seg['text']}\"")
        
        # Step 1: Run Devanagari Translation Loop via local LLM engine
        llm_payload = query_local_translation_engine(seg["text"], seg["original_window_ms"])
        print(f"      -> Target Devanagari: \"{llm_payload.translated_text}\" (Strategy: {llm_payload.syllable_strategy.upper()})")
        
        # Step 2: Synthesize Native Human Pace 1.0x Audio Clip
        chunk_vocal_path = os.path.join(CHUNKS_DIR, f"vocal_chunk_{idx}.wav")
        generated_asset_path = generate_unwarped_voice_piper(llm_payload.translated_text, chunk_vocal_path)
        
        if not generated_asset_path or os.path.getsize(generated_asset_path) == 0:
            print(f"      ⚠️ Voice generation failed for segment index #{idx}. Skipping payload serialization.")
            continue

        # Step 3: Parse Real Generated Audio Length metrics
        natural_duration_ms = extract_audio_duration_ms(generated_asset_path)
        print(f"      -> Realized Vocal Length: {natural_duration_ms}ms vs Target Window: {seg['original_window_ms']}ms")
        
        # Step 4: Run Strategic Matrix Calculations to pick timeline directives
        directives = app_manifest.compute_orchestration_directives(
            seg["original_window_ms"], 
            natural_duration_ms
        )
        print(f"      -> Assigned State Directive: {directives.timeline_action} (Delta: {directives.action_duration_ms}ms)")

        # Step 5: Construct the validated Playback Segment Packet data tree node
        node = app_manifest.PlaybackSegmentNode(
            segment_index=idx,
            source_text=seg["text"],
            target_text=llm_payload.translated_text,
            anchor_timestamps=app_manifest.AnchorTimestamps(
                start_ms=seg["start_ms"],
                end_ms=seg["end_ms"],
                original_window_ms=seg["original_window_ms"]
            ),
            audio_asset=app_manifest.AudioAssetMetadata(
                local_path=os.path.relpath(generated_asset_path, SCRIPT_ROOT),
                natural_duration_ms=natural_duration_ms
            ),
            orchestration=directives
        )
        playback_timeline_nodes.append(node)

    # Step 6: Generate and write the final application data contract manifest to disk
    master_manifest = app_manifest.LectureDubbingManifest(
        source_video_hash=payload["source_video_hash"],
        total_duration_seconds=payload["total_duration_seconds"],
        total_segments=len(playback_timeline_nodes),
        playback_timeline=playback_timeline_nodes
    )
    
    output_manifest_json = os.path.join(STUDIO_ASSETS, f"timeline_manifest_{payload['source_video_hash']}.json")
    master_manifest.serialize_manifest(output_manifest_json)
    
    print(f"\n✅ Phase 2 inference loop closed cleanly. Output data contract generated successfully.")
    return output_manifest_json

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_inference_queue(sys.argv[1])
    else:
        print("Usage error: Run script passing path parameter target checkpoint file.")