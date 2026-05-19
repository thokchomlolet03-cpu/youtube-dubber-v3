import os
import sys
import json
import subprocess

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
STUDIO_ASSETS = os.path.join(SCRIPT_ROOT, "studio_assets")

def build_simulation_filter_graph(manifest_data: dict, no_vocals_path: str) -> tuple:
    filter_nodes = []
    audio_inputs = []
    audio_input_counter = 2
    
    video_segments_out = []
    vocal_segments_out = []
    ambient_segments_out = []

    timeline = manifest_data["playback_timeline"]
    total_segments = len(timeline)

    if total_segments == 0:
        raise ValueError("Cannot compile viewport matrix for an empty playback timeline array.")

    # Up-front architectural pre-splitter mapping pass
    bg_split_labels = "".join([f"[bg_split_{i}]" for i in range(total_segments)])
    filter_nodes.append(f"[1:a]asplit=outputs={total_segments}{bg_split_labels}")

    for i, node in enumerate(timeline):
        idx = node["segment_index"]
        start_sec = node["anchor_timestamps"]["start_ms"] / 1000.0
        end_sec = node["anchor_timestamps"]["end_ms"] / 1000.0
        action = node["orchestration"]["timeline_action"]
        action_delta_sec = node["orchestration"]["action_duration_ms"] / 1000.0
        
        # 1. Video Track Window Slicing Engine
        video_trim_label = f"[v_trim_{idx}]"
        filter_nodes.append(f"[0:v]trim=start={start_sec:.4f}:end={end_sec:.4f},setpts=PTS-STARTPTS{video_trim_label}")
        if action == "FREEZE_HOLD":
            video_frozen_label = f"[v_out_{idx}]"
            filter_nodes.append(f"{video_trim_label}tpad=stop_duration={action_delta_sec:.4f}:stop_mode=clone{video_frozen_label}")
            video_segments_out.append(video_frozen_label)
        else:
            video_segments_out.append(video_trim_label)

        # 2. Vocal Stream Normalization (Forces all inputs to perfect Stereo 44100Hz)
        vocal_raw_label = f"[{audio_input_counter}:a]"
        vocal_resampled_label = f"[vocal_resamp_{idx}]"
        audio_inputs.append(os.path.join(SCRIPT_ROOT, node["audio_asset"]["local_path"]))
        audio_input_counter += 1

        # High-Fidelity Resampling Node Injection Pass
        filter_nodes.append(f"{vocal_raw_label}aresample=osr=44100:ochl=stereo{vocal_resampled_label}")

        if action == "PAD_EMPTY":
            vocal_padded_label = f"[vocal_out_{idx}]"
            filter_nodes.append(f"{vocal_resampled_label}apad=pad_dur={action_delta_sec:.4f}{vocal_padded_label}")
            vocal_segments_out.append(vocal_padded_label)
        else:
            vocal_segments_out.append(vocal_resampled_label)

        # 3. Ambient Background Track Slicing Loop
        bg_cursor_input = f"[bg_split_{i}]"
        bg_trim_label = f"[bg_trim_{idx}]"
        filter_nodes.append(f"{bg_cursor_input}atrim=start={start_sec:.4f}:end={end_sec:.4f},asetpts=PTS-STARTPTS{bg_trim_label}")
        
        bg_extended_label = f"[bg_ext_{idx}]"
        if action == "FREEZE_HOLD":
            filter_nodes.append(f"{bg_trim_label}apad=pad_dur={action_delta_sec:.4f}{bg_extended_label}")
        else:
            bg_extended_label = bg_trim_label
            
        ambient_segments_out.append(bg_extended_label)

    # Macro Track Concatenation Pass
    concat_video_inputs = "".join(video_segments_out)
    filter_nodes.append(f"{concat_video_inputs}concat=n={total_segments}:v=1:a=0[master_video_track]")

    concat_ambient_inputs = "".join(ambient_segments_out)
    filter_nodes.append(f"{concat_ambient_inputs}concat=n={total_segments}:v=0:a=1[raw_ambient_track]")

    # Vocal OLA Overlap crossfading engine pass
    if total_segments == 1:
        filter_nodes.append(f"{vocal_segments_out[0]}anull[compiled_vocals]")
    else:
        current_vocal_chain_head = vocal_segments_out[0]
        for i in range(1, total_segments):
            next_vocal_segment = vocal_segments_out[i]
            
            prev_node_is_gap = timeline[i-1]["is_gap"]
            curr_node_is_gap = timeline[i]["is_gap"]
            
            xfade_buffer_label = f"[v_chain_{i}]"
            target_out_label = "[compiled_vocals]" if i == (total_segments - 1) else xfade_buffer_label
            
            if prev_node_is_gap or curr_node_is_gap:
                filter_nodes.append(f"{current_vocal_chain_head}{next_vocal_segment}concat=n=2:v=0:a=1{target_out_label}")
            else:
                # Optimized configuration targeting total compliance across FFmpeg 8.x environments
                filter_nodes.append(f"{current_vocal_chain_head}{next_vocal_segment}acrossfade=d=0.08:c1=qsin:c2=qsin{target_out_label}")
            
            current_vocal_chain_head = xfade_buffer_label

    # 4. Parametric Dynamic Sidechain Ducking Engine Implementation Pass
    # Split vocals to use as a dynamic volume envelope controller track
    filter_nodes.append("[compiled_vocals]asplit=2[vocal_playback_track][vocal_sidechain_key]")
    
    # Process ambient audio track based on vocal energy levels
    filter_nodes.append(
        "[raw_ambient_track][vocal_sidechain_key]sidechaincompress=threshold=0.15:ratio=4:attack=100:release=300[ducked_ambient_track]"
    )

    # Master Mixing and Normalization Pass
    filter_nodes.append("[vocal_playback_track][ducked_ambient_track]amix=inputs=2:duration=longest:normalize=0[mixed_audio_sum]")
    filter_nodes.append("[mixed_audio_sum]loudnorm=I=-16:TP=-1.5:LRA=11[master_audio_track]")

    return ";".join(filter_nodes), audio_inputs

def emulate_player_playback(manifest_json_path: str, output_video_path: str):
    print(f"\n[PHASE 3] Launching Elastic Playback Emulator Core Simulator...")
    with open(manifest_json_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    video_hash = manifest_data["source_video_hash"]
    checkpoint_path = os.path.join(STUDIO_ASSETS, f"ingestion_checkpoint_{video_hash}.json")
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)
        
    source_video = checkpoint_data["source_video_path"]
    no_vocals_wav = checkpoint_data["no_vocals_wav_path"]

    filter_graph, dynamic_audio_inputs = build_simulation_filter_graph(manifest_data, no_vocals_wav)

    cmd = ["ffmpeg", "-y", "-i", source_video, "-i", no_vocals_wav]
    for audio_path in dynamic_audio_inputs:
        cmd.extend(["-i", audio_path])

    cmd.extend([
        "-filter_complex", filter_graph, "-map", "[master_video_track]", "-map", "[master_audio_track]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", output_video_path
    ])

    print("   -> Compiling non-destructive media matrix. Processing timeline holds...")
    try:
        res = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"\n=======================================================")
        print("🏁 SIMULATOR PLAYBACK COMPLETE: MOBILE EXPERIENCE RENDERED")
        print(f"📂 High-Fidelity Output Location: {output_video_path}")
        print("=======================================================\n")
    except subprocess.CalledProcessError as e:
        err_log = e.stderr.decode().strip() if e.stderr else "Unknown FFmpeg Console Panic"
        print(f"\n❌ Playback Emulator Core Exception:\n{err_log}")
        raise e