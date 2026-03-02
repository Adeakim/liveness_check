import os
import tempfile

import cv2
import numpy as np
from fastapi import UploadFile

from app.config import get_settings

settings = get_settings()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


async def load_image_from_upload(upload: UploadFile) -> np.ndarray | None:
    """
    Load an RGB image from an uploaded file (JPEG, PNG, WebP).

    Returns:
        RGB numpy array (H, W, 3), or None if decoding fails.
    """
    try:
        contents = await upload.read()
        np_arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def is_image_content_type(content_type: str | None) -> bool:
    return content_type in ALLOWED_IMAGE_TYPES if content_type else False


def is_video_content_type(content_type: str | None) -> bool:
    return content_type in {"video/mp4", "video/quicktime", "video/x-msvideo"} if content_type else False


async def extract_frames_from_video(
    video_file: UploadFile, num_frames: int | None = None
) -> list[np.ndarray]:
    """
    Extracts evenly spaced frames from an uploaded video file.

    Args:
        video_file: FastAPI UploadFile object.
        num_frames: Number of frames to extract. Uses config value if not specified.

    Returns:
        List of RGB numpy arrays, or empty list if extraction fails.
    """
    if num_frames is None:
        num_frames = settings.video_num_frames

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=settings.video_temp_suffix) as temp_file:
            temp_path = temp_file.name
            contents = await video_file.read()
            temp_file.write(contents)

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        frames = []
        indices = np.linspace(0, total_frames - 2, num_frames, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                if w > h:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames

    except Exception:
        return []
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
