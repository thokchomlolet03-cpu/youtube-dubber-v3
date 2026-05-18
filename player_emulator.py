import os
import sys
import json
import subprocess

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
STUDIO_ASSETS = os.path.join(SCRIPT_ROOT, "studio_assets")

def build_simulation_filter_graph(manifest_data: dict, no_vocals_path: str) -> tuple:
    """
    Translates the v3.1 Elastic Timeline specification into a frame-accurate 
    FFmpeg complex filter graph configuration string.
    """
    video_inputs = []
    audio_inputs = []
    
    filter_nodes = []
    
    # Track the active file indices inside our composite multimedia mix matrix
    # Input 0 = Source Video, Input 1 = Background Ambiance (no_vocals.wav)
    # Incoming segment audio assets occupy indices 2, 3, 4... N+2
    audio_input_counter = 2
    
    video_segments_out = []
    vocal_segments_out = []
    ambient_segments_out = []

    timeline = manifest_data["playback_timeline"]

    for node in timeline:
        idx = node["segment_index"]
        start_sec = node["anchor_timestamps"]["start_ms"] / 1000.0
        end_sec = node["anchor_timestamps"]["end_ms"] / 1000.0
        orig_dur_sec = node["anchor_timestamps"]["original_window_ms"] / 1000.0
        action = node["orchestration"]["timeline_action"]
        action_delta_sec = node["orchestration"]["action_duration_ms"] / 1000.0
        
        # 🎬 SECTION A: NON-DESTRUCTIVE VIDEO TIMELINE MANIPULATION
        # Isolate the precise frame window corresponding to this lecture topic slice
        video_label = f"v_trim_{idx}"
        filter_nodes.append(
            f"[0:v]trim=start={start_sec:.4f}:end={end_sec:.4f},setpts=PTS-STARTPTS[{video_label}]"
        )
        
        if action == "FREEZE_HOLD":
            # State FREEZE_HOLD: Clone the final video frame buffer to let the translation breathe
            # tpad clone extends the final frame of the video segment cleanly without draining memory
            frozen_video_label = f"v_frozen_{idx}"
            filter_nodes.append(
                f"[{video_label}]tpad=stop_duration={action_delta_sec:.4f}:stop_mode=clone[{frozen_video_label}]"
            )
            video_segments_out.append(f"[{frozen_video_label}]")
        else:
            # States NORMAL_SYNC & PAD_EMPTY: Stream frames 1:1 with regular playback speed
            video_segments_out.append(f"[{video_label}]")

        # 🎙️ SECTION B: VOCAL STREAM ORCHESTRATION & SILENCE PADDING
        vocal_label = f"vocal_stream_{idx}"
        if action == "PAD_EMPTY":
            # State PAD_EMPTY: Keep true human cadence and fill empty space with trailing silence
            filter_nodes.append(
                f"[{audio_input_counter}:a]apad=pad_dur={action_delta_sec:.4f}[{vocal_label}]"
            )
        else:
            filter_nodes.append(
                f"[{audio_input_counter}:a]asplit=1[{vocal_label}]"
            )
        vocal_segments_out.append(f"[{vocal_label}]")
        audio_inputs.append(os.path.join(SCRIPT_ROOT, node["audio_asset"]["local_path"]))
        audio_input_counter += 1

        # 🎚️ SECTION C: PARAMETRIC FREQUENCY-SELECTIVE SIDECHAIN DUCKING
        # Isolate the background ambiance portion mapping to this segment's window
        bg_trim_label = f"bg_trim_{idx}"
        filter_nodes.append(
            f"[1:a]atrim=start={start_sec:.4f}:end={end_sec:.4f},asetpts=PTS-STARTPTS[{bg_trim_label}]"
        )
        
        # Apply parametric multi-band filtering to protect vocal format intelligibility
        # We attenuate only the 1kHz-3.5kHz pocket, keeping base room hum completely stable
        ducked_bg_label = f"bg_ducked_{idx}"
        duck_db = node["orchestration"]["ambient_ducking_profile"]["duck_attenuation_db"]
        
        # If video is frozen, we extend the background noise clip to match the expanded timeline layout
        bg_paddable_label = f"bg_pad_{idx}"
        if action == "FREEZE_HOLD":
            filter_nodes.append(
                f"[{bg_trim_label}]apad=pad_dur={action_delta_sec:.4f}[{bg_paddable_label}]"
            )
        else:
            filter_nodes.append(f"[{bg_trim_label}]asplit=1[{bg_paddable_label}]")

        filter_nodes.append(
            f"[{bg_paddable_label}]equalizer=f=2250:width_type=h:width=1250:g={duck_db:.1f}[{ducked_bg_label}]"
        )
        ambient_segments_out.append(f"[{ducked_bg_label}]")

    # 🔗 SECTION D: MASTER CONCATENATION & MULTI-STREAM COMPILE
    total_video_nodes = len(video_segments_out)
    concat_video_nodes = "".join(video_segments_out)
    filter_nodes.append(f"{concat_video_nodes}concat=n={total_video_nodes}:v=1:a=0[master_video_track]")

    total_vocal_nodes = len(vocal_segments_out)
    concat_vocal_nodes = "".join(vocal_segments_out)
    filter_nodes.append(f"{concat_vocal_nodes}concat=n={total_vocal_nodes}:v=0:a=1[compiled_vocals]")

    total_ambient_nodes = len(ambient_segments_out)
    concat_ambient_nodes = "".join(ambient_segments_out)
    filter_nodes.append(f"{concat_ambient_nodes}concat=n={total_ambient_nodes}:v=0:a=1[compiled_ambience]")

    # Accumulate and summer both audio layers together cleanly with normalization protections disabled
    filter_nodes.append("[compiled_vocals][compiled_ambience]amix=inputs=2:duration=longest:normalize=0[master_audio_track]")

    full_filter_graph = ";".join(filter_nodes)
    return full_filter_graph, audio_inputs

def emulate_player_playback(manifest_json_path: str, output_video_path: str):
    """
    Reads the data contract manifest and executes an advanced FFmpeg processing pass
    to render an exact high-fidelity simulation of the native mobile app experience.
    """
    print(f"\n[PHASE 3] Launching Elastic Playback Emulator Core Simulator...")
    
    with open(manifest_json_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Resolve primary static source file parameters saved in our assets directory
    video_hash = manifest_data["source_video_hash"]
    checkpoint_path = os.path.join(STUDIO_ASSETS, f"ingestion_checkpoint_{video_hash}.json")
    
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)
        
    source_video = checkpoint_data["source_video_path"]
    no_vocals_wav = checkpoint_data["no_vocals_wav_path"]

    print("   -> Assembling multi-stream matrix components and parametric filter curves...")
    filter_graph, dynamic_audio_inputs = build_simulation_filter_graph(manifest_data, no_vocals_wav)

    # Construct the structural execution parameters for our non-destructive media rendering command
    cmd = [
        "ffmpeg", "-y",
        "-i", source_video,     # Input 0
        "-i", no_vocals_wav     # Input 1
    ]
    
    # Append the sequence of localized 1.0x human pace local voice stems dynamically
    for audio_path in dynamic_audio_inputs:
        cmd.extend(["-i", audio_path])

    # Map the generated graph and enforce professional broadcast loudness leveling targets (-16 LUFS)
    cmd.extend([
        "-filter_complex", filter_graph,
        "-map", "[master_video_track]",
        "-map", "[master_audio_track]",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", 
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        output_video_path
    ])

    print("   -> Compiling non-destructive media matrix. Processing timeline holds...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"\n=======================================================")
        print("🏁 SIMULATOR PLAYBACK COMPLETE: MOBILE EXPERIENCED RENDERED")
        print(f"📂 High-Fidelity Output Location: {output_video_path}")
        print("=======================================================\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Playback Emulator Core Exception. FFmpeg processing error code: {e.returncode}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        emulate_player_playback(sys.argv[1], sys.argv[2])
    else:
        print("Usage error: Run script passing manifest target path and desired output filename variables.")