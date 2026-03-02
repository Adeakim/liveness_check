from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Liveness detection
    liveness_min_valid_frames: int = 2
    liveness_center_ratio_min: float = 0.5
    liveness_center_ratio_max: float = 2.0
    liveness_left_turn_threshold: float = 0.50
    liveness_mirror_threshold: float = 1.5
    liveness_mouth_open_threshold: float = 0.03  # Min vertical mouth distance (normalized) to count as "open"
    face_detection_confidence: float = 0.3
    face_match_tolerance: float = 0.6  # Lower = stricter (face_recognition default)

    # Video processing
    video_num_frames: int = 4
    video_max_size_mb: int = 50
    video_temp_suffix: str = ".mp4"

    class Config:
        env_prefix = ""


def get_settings() -> Settings:
    return Settings()
