from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models import (
    CompareResponse,
    FaceMatchResult,
    LivenessResponse,
    LivenessResult,
)
from app.services.face_comparison import (
    compare_faces,
    get_best_face_encoding_from_frames,
    get_face_encoding_from_image,
)
from app.services.liveness import (
    check_liveness,
    check_liveness_head_turn,
    check_liveness_mouth,
)
from app.services.video_utils import (
    extract_frames_from_video,
    is_image_content_type,
    is_video_content_type,
    load_image_from_upload,
)

router = APIRouter(prefix="/liveness", tags=["liveness"])

ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo"}
ALLOWED_REFERENCE_TYPES = ALLOWED_VIDEO_TYPES | {"image/jpeg", "image/png", "image/webp"}


def _validate_video_content_type(content_type: str | None) -> None:
    if content_type and content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Use MP4. Got: {content_type}",
        )


@router.post("/check", response_model=LivenessResponse)
async def liveness_check_all(video: UploadFile = File(..., description="MP4 video")):
    """
    **Combined liveness check** — Head turn + mouth open.

    **Instructions:**
    1. Record 4–5 seconds
    2. Start looking straight at the camera
    3. Slowly turn your head to the left
    4. Open your mouth at some point (e.g. say "ah")
    5. Good lighting, face visible throughout
    """
    _validate_video_content_type(video.content_type)
    try:
        frames = await extract_frames_from_video(video)
        if not frames:
            return LivenessResponse(
                status="error",
                error="Could not extract frames from video. Ensure the file is a valid MP4 with at least 2 frames.",
            )
        passed, message, details = check_liveness(frames)
        return LivenessResponse(
            status="success" if passed else "failed",
            liveness=LivenessResult(passed=passed, message=message, details=details),
        )
    except HTTPException:
        raise
    except Exception as e:
        return LivenessResponse(status="error", error=str(e))


@router.post("/check/head-turn", response_model=LivenessResponse)
async def liveness_check_head_turn(video: UploadFile = File(..., description="MP4 video")):
    """
    **Head turn only** — Center → left movement.

    **Instructions:**
    1. Record 4–5 seconds
    2. Start looking straight at the camera
    3. Slowly turn your head to the left
    """
    _validate_video_content_type(video.content_type)
    try:
        frames = await extract_frames_from_video(video)
        if not frames:
            return LivenessResponse(
                status="error",
                error="Could not extract frames from video. Ensure the file is a valid MP4 with at least 2 frames.",
            )
        passed, message, details = check_liveness_head_turn(frames)
        return LivenessResponse(
            status="success" if passed else "failed",
            liveness=LivenessResult(passed=passed, message=message, details=details),
        )
    except HTTPException:
        raise
    except Exception as e:
        return LivenessResponse(status="error", error=str(e))


@router.post("/check/mouth", response_model=LivenessResponse)
async def liveness_check_mouth(video: UploadFile = File(..., description="MP4 video")):
    """
    **Mouth open only** — Open your mouth during the video.

    **Instructions:**
    1. Record 4–5 seconds
    2. Face the camera
    3. Open your mouth at some point (e.g. say "ah")
    """
    _validate_video_content_type(video.content_type)
    try:
        frames = await extract_frames_from_video(video)
        if not frames:
            return LivenessResponse(
                status="error",
                error="Could not extract frames from video. Ensure the file is a valid MP4 with at least 2 frames.",
            )
        passed, message, details = check_liveness_mouth(frames)
        return LivenessResponse(
            status="success" if passed else "failed",
            liveness=LivenessResult(passed=passed, message=message, details=details),
        )
    except HTTPException:
        raise
    except Exception as e:
        return LivenessResponse(status="error", error=str(e))


def _validate_reference_content_type(content_type: str | None) -> None:
    if content_type and content_type not in ALLOWED_REFERENCE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reference type. Use image (JPEG/PNG/WebP) or video (MP4). Got: {content_type}",
        )


@router.post("/compare", response_model=CompareResponse)
async def liveness_compare(
    video: UploadFile = File(..., description="Liveness video (MP4)"),
    reference: UploadFile = File(..., description="Reference face — image (JPEG/PNG/WebP) or video (MP4)"),
):
    """
    **Liveness + face match** — Verify the person in the liveness video matches the reference.

    Runs full liveness check (head turn + mouth open) and compares the face to a
    reference photo or video. Passes only if both succeed.

    **Instructions:**
    1. Liveness video: same as /check (head turn + mouth open)
    2. Reference: photo or short video of the person to match (e.g. ID photo)
    """
    _validate_video_content_type(video.content_type)
    _validate_reference_content_type(reference.content_type)

    try:
        liveness_frames = await extract_frames_from_video(video)
        if not liveness_frames:
            return CompareResponse(
                status="error",
                passed=False,
                error="Could not extract frames from video. Ensure the file is a valid MP4.",
            )

        # Load reference image (from file or video frame)
        reference_rgb = None
        if is_image_content_type(reference.content_type):
            reference_rgb = await load_image_from_upload(reference)
        elif is_video_content_type(reference.content_type):
            ref_frames = await extract_frames_from_video(reference)
            if ref_frames:
                # Use middle frame for best face visibility
                reference_rgb = ref_frames[len(ref_frames) // 2]

        if reference_rgb is None:
            return CompareResponse(
                status="error",
                passed=False,
                error="Could not load reference. Ensure it contains a clear, visible face.",
            )

        # Liveness check
        liveness_passed, liveness_msg, liveness_details = check_liveness(liveness_frames)

        # Face comparison
        ref_encoding = get_face_encoding_from_image(reference_rgb)
        if ref_encoding is None:
            return CompareResponse(
                status="error",
                passed=False,
                liveness=LivenessResult(passed=liveness_passed, message=liveness_msg, details=liveness_details),
                face_match=FaceMatchResult(
                    matched=False,
                    distance=0.0,
                    message="No face detected in reference.",
                ),
                error="No face detected in reference image/video.",
            )

        video_encoding = get_best_face_encoding_from_frames(liveness_frames)
        if video_encoding is None:
            return CompareResponse(
                status="failed",
                passed=False,
                liveness=LivenessResult(passed=liveness_passed, message=liveness_msg, details=liveness_details),
                face_match=FaceMatchResult(
                    matched=False,
                    distance=0.0,
                    message="No face detected in liveness video.",
                ),
            )

        matched, distance = compare_faces(ref_encoding, video_encoding)
        face_match_msg = "Face matches reference." if matched else f"Face does not match reference (distance: {distance:.3f})."

        passed = liveness_passed and matched
        return CompareResponse(
            status="success" if passed else "failed",
            passed=passed,
            liveness=LivenessResult(passed=liveness_passed, message=liveness_msg, details=liveness_details),
            face_match=FaceMatchResult(matched=matched, distance=round(distance, 4), message=face_match_msg),
        )
    except HTTPException:
        raise
    except Exception as e:
        return CompareResponse(status="error", passed=False, error=str(e))
