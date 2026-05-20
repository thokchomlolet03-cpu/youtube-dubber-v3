import os
import sys
import json
import subprocess

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
STUDIO_ASSETS = os.path.join(SCRIPT_ROOT, "studio_assets")

def generate_mobile_media3_composition(manifest_data: dict, no_vocals_path: str) -> dict:
    """
    Transforms the multi-input desktop filter graph into a clean, event-driven 
    Android JSON data contract to support O(1) memory dynamic media playback.
    """
    timeline = manifest_data["playback_timeline"]
    
    mobile_composition_payload = {
        "target_mobile_specification": "Media3-Scaffold-v1.2",
        "source_fingerprint_hash": manifest_data["source_video_hash"],
        "base_background_audio_track": os.path.relpath(no_vocals_path, SCRIPT_ROOT),
        "target_vocal_sample_rate_hz": 44100,
        "target_vocal_channel_layout": "stereo",
        "composition_playlist": []
    }

    for node in timeline:
        idx = node["segment_index"]
        action = node["orchestration"]["timeline_action"]
        
        segment_descriptor = {
            "segment_index": idx,
            "is_gap_interval": node["is_gap"],
            "source_english_text": node["source_text"],
            "target_hindi_text": node["target_text"],
            "vocal_asset_local_path": node["audio_asset"]["local_path"],
            "physical_media_bounds": {
                "start_ms": node["anchor_timestamps"]["start_ms"],
                "end_ms": node["anchor_timestamps"]["end_ms"],
                "duration_ms": node["anchor_timestamps"]["original_window_ms"]
            },
            "elastic_player_directives": {
                "timeline_action": action,
                "action_delta_ms": node["orchestration"]["action_duration_ms"],
                "cumulative_drift_offset_ms": node["orchestration"]["accumulated_drift_offset_ms"]
            },
            "parametric_vocal_audio_effects": {
                "resample_frequency_hz": 44100,
                "channel_layout_target": "stereo",
                "apply_dynamic_sidechain_ducking": not node["is_gap"],
                "ducking_attenuation_db": node["orchestration"]["ambient_ducking_profile"]["duck_attenuation_db"],
                "envelope_attack_ms": node["orchestration"]["ambient_ducking_profile"]["attack_ms"],
                "envelope_release_ms": node["orchestration"]["ambient_ducking_profile"]["release_ms"]
            }
        }
        
        mobile_composition_payload["composition_playlist"].append(segment_descriptor)

    return mobile_composition_payload

def emulate_player_playback(manifest_json_path: str, output_video_path: str):
    """
    Maintains complete fallback desktop verification compatibility while 
    automatically exporting the native mobile composition layout metadata payload.
    """
    print(f"\n[PHASE 3] Launching Elastic Playback Emulator Core Simulator...")
    with open(manifest_json_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    video_hash = manifest_data["source_video_hash"]
    checkpoint_path = os.path.join(STUDIO_ASSETS, f"ingestion_checkpoint_{video_hash}.json")
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)
        
    source_video = checkpoint_data["source_video_path"]
    no_vocals_wav = checkpoint_data["no_vocals_wav_path"]

    # 1. Compile the Event-Driven Android Mobile Composition Manifest File
    mobile_payload = generate_mobile_media3_composition(manifest_data, no_vocals_wav)
    mobile_json_output_path = os.path.join(STUDIO_ASSETS, f"android_media3_composition_{video_hash}.json")
    
    with open(mobile_json_output_path, "w", encoding="utf-8") as mobile_f:
        json.dump(mobile_payload, mobile_f, ensure_ascii=False, indent=2)
    print(f"   -> [Mobile Porting] Exported native dynamic Media3 contract: {mobile_json_output_path}")

    # 2. Build desktop verification graph to ensure ongoing signal-flow correctness
    print("   -> Assembling verification multi-stream matrix components...")
    filter_nodes = []
    audio_inputs = []  # ◄── This is the valid local tracking list container variable
    audio_input_counter = 2
    video_segments_out = []
    vocal_segments_out = []
    ambient_segments_out = []

    timeline = manifest_data["playback_timeline"]
    total_segments = len(timeline)

    bg_split_labels = "".join([f"[bg_split_{i}]" for i in range(total_segments)])
    filter_nodes.append(f"[1:a]asplit=outputs={total_segments}{bg_split_labels}")

    for i, node in enumerate(timeline):
        idx = node["segment_index"]
        start_sec = node["anchor_timestamps"]["start_ms"] / 1000.0
        end_sec = node["anchor_timestamps"]["end_ms"] / 1000.0
        action = node["orchestration"]["timeline_action"]
        action_delta_sec = node["orchestration"]["action_duration_ms"] / 1000.0
        
        video_trim_label = f"[v_trim_{idx}]"
        filter_nodes.append(f"[0:v]trim=start={start_sec:.4f}:end={end_sec:.4f},setpts=PTS-STARTPTS{video_trim_label}")
        if action == "FREEZE_HOLD":
            video_frozen_label = f"[v_out_{idx}]"
            filter_nodes.append(f"{video_trim_label}tpad=stop_duration={action_delta_sec:.4f}:stop_mode=clone{video_frozen_label}")
            video_segments_out.append(video_frozen_label)
        else:
            video_segments_out.append(video_trim_label)

        vocal_raw_label = f"[{audio_input_counter}:a]"
        vocal_resampled_label = f"[vocal_resamp_{idx}]"
        audio_inputs.append(os.path.join(SCRIPT_ROOT, node["audio_asset"]["local_path"]))
        audio_input_counter += 1

        filter_nodes.append(f"{vocal_raw_label}aresample=osr=44100:ochl=stereo{vocal_resampled_label}")

        if action == "PAD_EMPTY":
            vocal_padded_label = f"[vocal_out_{idx}]"
            filter_nodes.append(f"{vocal_resampled_label}apad=pad_dur={action_delta_sec:.4f}{vocal_padded_label}")
            vocal_segments_out.append(vocal_padded_label)
        else:
            vocal_segments_out.append(vocal_resampled_label)

        bg_cursor_input = f"[bg_split_{i}]"
        bg_trim_label = f"[bg_trim_{idx}]"
        filter_nodes.append(f"{bg_cursor_input}atrim=start={start_sec:.4f}:end={end_sec:.4f},asetpts=PTS-STARTPTS{bg_trim_label}")
        
        bg_extended_label = f"[bg_ext_{idx}]"
        if action == "FREEZE_HOLD":
            filter_nodes.append(f"{bg_trim_label}apad=pad_dur={action_delta_sec:.4f}{bg_extended_label}")
        else:
            bg_extended_label = bg_trim_label
        ambient_segments_out.append(bg_extended_label)

    concat_video_inputs = "".join(video_segments_out)
    filter_nodes.append(f"{concat_video_inputs}concat=n={total_segments}:v=1:a=0[master_video_track]")

    concat_ambient_inputs = "".join(ambient_segments_out)
    filter_nodes.append(f"{concat_ambient_inputs}concat=n={total_segments}:v=0:a=1[raw_ambient_track]")

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
            filter_nodes.append(f"{current_vocal_chain_head}{next_vocal_segment}acrossfade=d=0.08:c1=qsin:c2=qsin{target_out_label}")
        current_vocal_chain_head = xfade_buffer_label

    filter_nodes.append("[compiled_vocals]asplit=2[vocal_playback_track][vocal_sidechain_key]")
    filter_nodes.append("[raw_ambient_track][vocal_sidechain_key]sidechaincompress=threshold=0.15:ratio=4:attack=100:release=300[ducked_ambient_track]")
    filter_nodes.append("[vocal_playback_track][ducked_ambient_track]amix=inputs=2:duration=longest:normalize=0[mixed_audio_sum]")
    filter_nodes.append("[mixed_audio_sum]loudnorm=I=-16:TP=-1.5:LRA=11[master_audio_track]")

    filter_graph = ";".join(filter_nodes)

    cmd = ["ffmpeg", "-y", "-i", source_video, "-i", no_vocals_wav]
    
    # FIXED: Iterate using the matching 'audio_inputs' list mapping pointer variable
    for audio_path in audio_inputs:
        cmd.extend(["-i", audio_path])

    cmd.extend([
        "-filter_complex", filter_graph, "-map", "[master_video_track]", "-map", "[master_audio_track]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", output_video_path
    ])

    print("   -> Running desktop verification render pass...")
    try:
        res = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"\n=======================================================")
        print("🏁 SIMULATOR PLAYBACK COMPLETE: DESKTOP RENDER PASS SECURED")
        print("=======================================================\n")
    except subprocess.CalledProcessError as e:
        err_log = e.stderr.decode().strip() if e.stderr else "Unknown FFmpeg Panic"
        print(f"\n❌ Playback Emulator Core Exception:\n{err_log}")
        raise e