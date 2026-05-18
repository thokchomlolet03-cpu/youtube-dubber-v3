import os
import json
from typing import Literal, List
from pydantic import BaseModel, Field, field_validator

class AnchorTimestamps(BaseModel):
    """Tracks raw timeline alignments captured directly from the original video track."""
    start_ms: int = Field(description="The starting point of the vocal segment in absolute container milliseconds.")
    end_ms: int = Field(description="The ending point of the vocal segment in absolute container milliseconds.")
    original_window_ms: int = Field(description="Calculated duration of the target English timeline window.")

    @field_validator('original_window_ms')
    @classmethod
    def verify_temporal_window(cls, v: int, info) -> int:
        """Enforces absolute timeline consistency invariants."""
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
    # State Machine Routing Options mapping directly to the v3.1 master routing engine matrix
    timeline_action: Literal['NORMAL_SYNC', 'PAD_EMPTY', 'FREEZE_HOLD'] = Field(...)
    action_duration_ms: int = Field(description="Calculated time padding factor applied directly to video frames or audio tails.")
    accumulated_drift_offset_ms: int = Field(default=0, description="Clock drift compensation tracking index variable.")
    ambient_ducking_profile: DuckingProfile = Field(default_factory=DuckingProfile)

class PlaybackSegmentNode(BaseModel):
    """The unified metadata packet representing a single event-driven timeline node."""
    segment_index: int = Field(description="Sequential timeline index marker tracking position.")
    source_text: str = Field(description="The original parsed text snippet extracted during the ASR loop.")
    target_text: str = Field(description="The localized Hinglish translation written explicitly in Devanagari characters.")
    anchor_timestamps: AnchorTimestamps
    audio_asset: AudioAssetMetadata
    orchestration: OrchestrationDirectives

class LectureDubbingManifest(BaseModel):
    """The master architectural source of truth connecting AI metrics to player logic."""
    manifest_version: str = Field(default="3.1.0", description="Strict tracking specification version.")
    source_video_hash: str = Field(description="Cryptographic file fingerprint ensuring static reference data bounds.")
    total_duration_seconds: float = Field(description="Absolute physical video track container duration.")
    total_segments: int = Field(description="Total count of independent timeline playlist nodes.")
    playback_timeline: List[PlaybackSegmentNode] = Field(default_factory=list)

    def serialize_manifest(self, output_json_path: str):
        """Writes the structured data layer cleanly to disk as a production-grade manifest contract."""
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        print(f"   -> [Manifest System] Successfully synchronized data contract: {output_json_path}")

    @classmethod
    def load_manifest(cls, input_json_path: str) -> "LectureDubbingManifest":
        """Loads and validates a structural timeline manifest straight into the application state."""
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

def compute_orchestration_directives(original_duration_ms: int, generated_duration_ms: int) -> OrchestrationDirectives:
    """
    Core Mathematical Routing Engine. Emulates the timeline state machine to map 
    relationships between speech lengths and video constraints without altering audio pitch.
    """
    # Calculate the speed ratio directly using millisecond tracking scales
    speed_ratio = generated_duration_ms / original_duration_ms
    
    # Check if the pacing metrics fit naturally within our strict human cadence boundaries
    if 0.85 <= speed_ratio <= 1.20:
        return OrchestrationDirectives(
            timeline_action="NORMAL_SYNC",
            action_duration_ms=0
        )
    elif speed_ratio < 0.85:
        # Case: Hindi audio is too short. Keep human cadence and pad the tail with trailing digital silence.
        silence_padding = original_duration_ms - generated_duration_ms
        return OrchestrationDirectives(
            timeline_action="PAD_EMPTY",
            action_duration_ms=silence_padding
        )
    else:
        # Case: Hindi audio runs over the window. Trigger an immediate video freeze-frame directive.
        freeze_duration = generated_duration_ms - original_duration_ms
        return OrchestrationDirectives(
            timeline_action="FREEZE_HOLD",
            action_duration_ms=freeze_duration
        )

if __name__ == "__main__":
    # Internal validation testing block to verify math logic compliance across edge states
    print("🔬 Running core manifest data engine validation tests...")
    test_sync = compute_orchestration_directives(4000, 4200)
    assert test_sync.timeline_action == "NORMAL_SYNC"
    
    test_pad = compute_orchestration_directives(4000, 2500)
    assert test_pad.timeline_action == "PAD_EMPTY"
    assert test_pad.action_duration_ms == 1500
    
    test_freeze = compute_orchestration_directives(4000, 6500)
    assert test_freeze.timeline_action == "FREEZE_HOLD"
    assert test_freeze.action_duration_ms == 2500
    print("✅ All validation testing routines exited successfully.")