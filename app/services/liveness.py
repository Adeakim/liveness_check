import os
import urllib.request
from typing import Tuple

import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core import image as mp_image

from app.config import get_settings

settings = get_settings()

# MediaPipe Face Landmarker model (Tasks API - works with mediapipe 0.10.31+)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Landmark indices (same topology as Face Mesh): nose, ears, mouth
NOSE_IDX = 1
LEFT_EAR_IDX = 234
RIGHT_EAR_IDX = 454
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14

_landmarker: mp_vision.FaceLandmarker | None = None


def _get_landmarker() -> mp_vision.FaceLandmarker:
    global _landmarker
    if _landmarker is not None:
        return _landmarker

    model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)

    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
    )
    _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def _analyze_frame(frame_rgb: np.ndarray) -> Tuple[float | None, float | None]:
    """
    Analyzes a single frame for head yaw and mouth openness.

    Returns:
        Tuple of (yaw_ratio, mouth_openness). Either may be None if face not detected.
    """
    try:
        img = mp_image.Image(mp_image.ImageFormat.SRGB, np.ascontiguousarray(frame_rgb))
        landmarker = _get_landmarker()
        result = landmarker.detect(img)

        if not result.face_landmarks:
            return None, None

        landmarks = result.face_landmarks[0]
        w = frame_rgb.shape[1]

        nose = landmarks[NOSE_IDX]
        left_ear = landmarks[LEFT_EAR_IDX]
        right_ear = landmarks[RIGHT_EAR_IDX]
        upper_lip = landmarks[UPPER_LIP_IDX]
        lower_lip = landmarks[LOWER_LIP_IDX]

        # Head yaw ratio
        nose_x = nose.x * w
        left_ear_x = left_ear.x * w
        right_ear_x = right_ear.x * w
        dist_nose_to_left = abs(nose_x - left_ear_x)
        dist_nose_to_right = abs(right_ear_x - nose_x)
        yaw_ratio = (dist_nose_to_left / dist_nose_to_right) if dist_nose_to_right != 0 else 999.0

        # Mouth openness: vertical distance between upper and lower lip (normalized 0-1)
        mouth_openness = abs(lower_lip.y - upper_lip.y)

        return yaw_ratio, mouth_openness
    except Exception:
        return None, None


def get_head_pose_yaw(frame_rgb: np.ndarray) -> float | None:
    """Convenience: returns only head yaw from frame analysis."""
    yaw, _ = _analyze_frame(frame_rgb)
    return yaw


def check_liveness(frames: list[np.ndarray]) -> Tuple[bool, str, dict]:
    """
    Validates liveness by checking for center-to-left head movement and mouth open.

    Requires user to start looking straight, turn head left, and open mouth.
    Detects photo/video replay attacks (static or mirrored).

    Returns:
        Tuple of (is_live: bool, message: str, details: dict).
    """
    ratios = []
    mouth_opennesses = []
    for frame in frames:
        r, mouth = _analyze_frame(frame)
        ratios.append(r)
        mouth_opennesses.append(mouth)

    valid_ratios = [r for r in ratios if r is not None]
    valid_mouths = [m for m in mouth_opennesses if m is not None]

    if len(valid_ratios) < settings.liveness_min_valid_frames:
        return False, "Face not detected clearly. Move slower and ensure good lighting.", {
            "ratios": ratios,
            "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
        }

    if max(valid_ratios) < settings.liveness_center_ratio_min:
        return False, "Start by looking straight at the camera.", {
            "ratios": ratios,
            "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
        }

    min_ratio = min(valid_ratios)
    max_ratio = max(valid_ratios)

    # Check head turn (center → left)
    head_turn_ok = min_ratio < settings.liveness_left_turn_threshold or max_ratio > settings.liveness_mirror_threshold
    if not head_turn_ok:
        return False, f"Head turn LEFT not detected. Range: {round(min_ratio, 2)} to {round(max_ratio, 2)}", {
            "min_ratio": round(min_ratio, 3),
            "max_ratio": round(max_ratio, 3),
            "ratios": [round(r, 2) if r is not None else None for r in ratios],
            "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
        }

    # Check mouth open in at least one frame
    mouth_open_ok = any(m >= settings.liveness_mouth_open_threshold for m in valid_mouths)
    if not mouth_open_ok:
        max_mouth = max(valid_mouths) if valid_mouths else 0
        return False, f"Open mouth not detected. Show your mouth open (e.g. say 'ah'). Max: {round(max_mouth, 3)}", {
            "min_ratio": round(min_ratio, 3),
            "max_ratio": round(max_ratio, 3),
            "ratios": [round(r, 2) if r is not None else None for r in ratios],
            "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
            "max_mouth_openness": round(max_mouth, 3),
        }

    return True, "Liveness verified (Center → Left, mouth open).", {
        "min_ratio": round(min_ratio, 3),
        "max_ratio": round(max_ratio, 3),
        "ratios": [round(r, 2) if r is not None else None for r in ratios],
        "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
    }


def check_liveness_head_turn(frames: list[np.ndarray]) -> Tuple[bool, str, dict]:
    """
    Validates liveness by checking for center-to-left head movement only.
    """
    ratios = []
    mouth_opennesses = []
    for frame in frames:
        r, mouth = _analyze_frame(frame)
        ratios.append(r)
        mouth_opennesses.append(mouth)

    valid_ratios = [r for r in ratios if r is not None]

    if len(valid_ratios) < settings.liveness_min_valid_frames:
        return False, "Face not detected clearly. Move slower and ensure good lighting.", {
            "ratios": ratios,
            "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
        }

    if max(valid_ratios) < settings.liveness_center_ratio_min:
        return False, "Start by looking straight at the camera.", {
            "ratios": ratios,
            "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
        }

    min_ratio = min(valid_ratios)
    max_ratio = max(valid_ratios)
    head_turn_ok = min_ratio < settings.liveness_left_turn_threshold or max_ratio > settings.liveness_mirror_threshold

    if head_turn_ok:
        return True, "Head turn verified (Center → Left).", {
            "min_ratio": round(min_ratio, 3),
            "max_ratio": round(max_ratio, 3),
            "ratios": [round(r, 2) if r is not None else None for r in ratios],
            "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
        }

    return False, f"Head turn LEFT not detected. Range: {round(min_ratio, 2)} to {round(max_ratio, 2)}", {
        "min_ratio": round(min_ratio, 3),
        "max_ratio": round(max_ratio, 3),
        "ratios": [round(r, 2) if r is not None else None for r in ratios],
        "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
    }


def check_liveness_mouth(frames: list[np.ndarray]) -> Tuple[bool, str, dict]:
    """
    Validates liveness by checking for mouth open only.
    """
    mouth_opennesses = []
    ratios = []
    for frame in frames:
        r, mouth = _analyze_frame(frame)
        ratios.append(r)
        mouth_opennesses.append(mouth)

    valid_mouths = [m for m in mouth_opennesses if m is not None]

    if len(valid_mouths) < settings.liveness_min_valid_frames:
        return False, "Face not detected clearly. Move slower and ensure good lighting.", {
            "ratios": ratios,
            "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
        }

    mouth_open_ok = any(m >= settings.liveness_mouth_open_threshold for m in valid_mouths)
    max_mouth = max(valid_mouths)

    if mouth_open_ok:
        return True, "Mouth open verified.", {
            "ratios": [round(r, 2) if r is not None else None for r in ratios],
            "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
            "max_mouth_openness": round(max_mouth, 3),
        }

    return False, f"Open mouth not detected. Show your mouth open (e.g. say 'ah'). Max: {round(max_mouth, 3)}", {
        "ratios": [round(r, 2) if r is not None else None for r in ratios],
        "mouth_opennesses": [round(m, 3) if m is not None else None for m in mouth_opennesses],
        "max_mouth_openness": round(max_mouth, 3),
    }
