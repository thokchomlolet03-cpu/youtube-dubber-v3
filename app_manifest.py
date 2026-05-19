import os
import json
import hashlib
from typing import Literal, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

class AnchorTimestamps(BaseModel):
    """Tracks raw timeline alignments captured directly from the original video track."""
    start_ms: int = Field(description="The starting point of the vocal segment in absolute container milliseconds.")
    end_ms: int = Field(description="The ending point of the vocal segment in absolute container milliseconds.")
    original_window_ms: int = Field(description="Calculated duration of the target English timeline window.")

    @field_validator('original_window_ms')
    @classmethod
    def verify_temporal_window(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("The target media execution window duration must be greater than 0ms.")
        return v

class AudioAssetMetadata(BaseModel):
    """Binds the state of pristine unwarped local speech clips generated in memory."""
    local_path: str = Field(description="System path to the unwarped local vocal asset.")
    natural_duration_ms: int = Field(description="The true runtime duration of the generated voice line at a 1.0x human pace.")
    sample_rate_hz: int = Field(default=44100, description="Pipeline standard frequency sample rate.")

class DuckingProfile(BaseModel):
    """Parametric configurations routing frequency-selective filter dampening layers."""
    duck_attenuation_db: float = Field(default=-12.0, description="Target attenuation applied exclusively to speech frequency pockets.")
    attack_ms: int = Field(default=100, description="Ramp-down interpolation speed constraint.")
    release_ms: int = Field(default=300, description="Gain restoration recovery duration tracking.")

class OrchestrationDirectives(BaseModel):
    """The playback state directives mapping language length directly to video coordinates."""
    timeline_action: Literal['NORMAL_SYNC', 'PAD_EMPTY', 'FREEZE_HOLD'] = Field(...)
    action_duration_ms: int = Field(description="Calculated time padding factor applied directly to video frames or audio tails.")
    accumulated_drift_offset_ms: int = Field(default=0, description="Calculated timeline displacement mapping up to this block.")
    ambient_ducking_profile: DuckingProfile = Field(default_factory=DuckingProfile)

class PlaybackSegmentNode(BaseModel):
    """The unified metadata packet representing a single event-driven timeline node."""
    segment_index: int = Field(description="Sequential timeline index marker tracking position.")
    source_text: str = Field(description="The original parsed text snippet extracted during the ASR loop.")
    target_text: str = Field(description="The localized Hinglish translation written explicitly in Devanagari characters.")
    is_gap: bool = Field(default=False, description="True if this node represents an ambient silence gap, not a speech segment.")
    anchor_timestamps: AnchorTimestamps
    audio_asset: AudioAssetMetadata
    orchestration: OrchestrationDirectives

class LectureDubbingManifest(BaseModel):
    """The master architectural source of truth connecting AI metrics to player logic."""
    model_config = ConfigDict(frozen=False)

    manifest_version: str = Field(default="3.2.0", description="Strict tracking specification version.")
    source_video_hash: str = Field(description="Cryptographic file fingerprint ensuring static reference data bounds.")
    total_duration_seconds: float = Field(description="Absolute physical video track container duration.")
    total_segments: int = Field(description="Total count of independent timeline playlist nodes.")
    playback_timeline: List[PlaybackSegmentNode] = Field(default_factory=list)
    virtual_timeline_map: List[Dict[str, Any]] = Field(default_factory=list)

    def verify_integrity(self, video_path: str) -> bool:
        if not os.path.exists(video_path):
            return False
        sha256_hash = hashlib.sha256()
        with open(video_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:12] == self.source_video_hash

    def compile_timeline_matrices(self):
        """
        Structures the virtual timeline coordinates for the player.
        NOTE (Finding N7): For PAD_EMPTY nodes, the virtual mapping segment cleanly 
        spans the full window duration without shifting downstream timestamps.
        """
        compiled_map = []

        for node in self.playback_timeline:
            drift = node.orchestration.accumulated_drift_offset_ms
            start_phys = node.anchor_timestamps.start_ms
            end_phys = node.anchor_timestamps.end_ms
            action = node.orchestration.timeline_action
            delta = node.orchestration.action_duration_ms
            
            compiled_map.append({
                "virtual_ms": start_phys + drift,
                "physical_ms": start_phys,
                "segment_index": node.segment_index,
                "timeline_action": action
            })
            
            if action == "FREEZE_HOLD":
                compiled_map.append({
                    "virtual_ms": end_phys + drift + delta,
                    "physical_ms": end_phys,
                    "segment_index": node.segment_index,
                    "timeline_action": "FREEZE_HOLD"
                })

        self.virtual_timeline_map = compiled_map

    def serialize_manifest(self, output_json_path: str):
        self.compile_timeline_matrices()
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        print(f"   -> [Manifest System] Synchronized updated data contract: {output_json_path}")

    @classmethod
    def load_manifest(cls, input_json_path: str) -> "LectureDubbingManifest":
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

def compute_orchestration_directives(original_duration_ms: int, generated_duration_ms: int, current_drift_ms: int) -> OrchestrationDirectives:
    if original_duration_ms <= 0:
        return OrchestrationDirectives(timeline_action="NORMAL_SYNC", action_duration_ms=0, accumulated_drift_offset_ms=current_drift_ms)
    speed_ratio = generated_duration_ms / original_duration_ms
    
    if 0.85 <= speed_ratio <= 1.20:
        return OrchestrationDirectives(timeline_action="NORMAL_SYNC", action_duration_ms=0, accumulated_drift_offset_ms=current_drift_ms)
    elif speed_ratio < 0.85:
        return OrchestrationDirectives(
            timeline_action="PAD_EMPTY", 
            action_duration_ms=(original_duration_ms - generated_duration_ms),
            accumulated_drift_offset_ms=current_drift_ms
        )
    else:
        return OrchestrationDirectives(
            timeline_action="FREEZE_HOLD", 
            action_duration_ms=(generated_duration_ms - original_duration_ms),
            accumulated_drift_offset_ms=current_drift_ms
        )