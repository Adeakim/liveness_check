"""
Face comparison service using face_recognition (dlib).
Compares face encodings from reference image/video with liveness video.
"""

from typing import Tuple

import face_recognition
import numpy as np

from app.config import get_settings

settings = get_settings()


def get_face_encoding_from_image(image_rgb: np.ndarray) -> np.ndarray | None:
    """
    Extract face encoding from an RGB image.

    Returns:
        128-dim encoding, or None if no face detected.
    """
    encodings = face_recognition.face_encodings(image_rgb)
    return encodings[0] if encodings else None


def compare_faces(
    reference_encoding: np.ndarray, candidate_encoding: np.ndarray
) -> Tuple[bool, float]:
    """
    Compare two face encodings.

    Returns:
        Tuple of (matched: bool, distance: float).
    """
    tolerance = settings.face_match_tolerance
    distance = float(face_recognition.face_distance([reference_encoding], candidate_encoding)[0])
    matched = distance <= tolerance
    return matched, distance


def get_best_face_encoding_from_frames(frames: list[np.ndarray]) -> np.ndarray | None:
    """
    Try to extract a face encoding from frames. Returns first successful encoding.
    """
    for frame in frames:
        encoding = get_face_encoding_from_image(frame)
        if encoding is not None:
            return encoding
    return None
