# Liveness Check API

FastAPI service for face liveness detection via head movement. Use this to verify a live person before approving transfers or other sensitive actions.

## How It Works

- **Input:** Short video (MP4) of the user's face
- **Detection:** MediaPipe analyzes head pose (yaw) and mouth openness across frames
- **Liveness:** User must start looking straight, turn head left, and open mouth at some point
- **Anti-spoofing:** Static photos/videos fail (no movement detected)

## Quick Start

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## API

All endpoints accept `multipart/form-data` with `video` (MP4 file).

### POST /liveness/check — Combined (head turn + mouth open)

Full liveness check: head turn **and** mouth open required.

### POST /liveness/check/head-turn — Head turn only

Head movement only: start looking straight, then turn left.

### POST /liveness/check/mouth — Mouth open only

Mouth openness only: open your mouth at some point in the video.

### POST /liveness/compare — Liveness + face match

Compares the person in the liveness video with a reference photo or video. Passes only if **both** liveness (head turn + mouth open) and face match succeed.

**Request:** `multipart/form-data` with:
- `video` — Liveness video (MP4)
- `reference` — Reference face: image (JPEG/PNG/WebP) or video (MP4)

---

**Response (check endpoints):**
```json
{
  "status": "success",
  "liveness": {
    "passed": true,
    "message": "Liveness verified (Center → Left, mouth open).",
    "details": {
      "min_ratio": 0.38,
      "max_ratio": 1.05,
      "ratios": [0.95, 1.02, 0.42, 0.38],
      "mouth_opennesses": [0.02, 0.04, 0.03, 0.05]
    }
  }
}
```

**Response (compare endpoint):**
```json
{
  "status": "success",
  "passed": true,
  "liveness": {
    "passed": true,
    "message": "Liveness verified (Center → Left, mouth open).",
    "details": { "min_ratio": 0.26, "max_ratio": 4.59, "ratios": [...], "mouth_opennesses": [...] }
  },
  "face_match": {
    "matched": true,
    "distance": 0.31,
    "message": "Face matches reference."
  }
}
```

### Example (curl)

```bash
# Combined check (all)
curl -X POST "http://localhost:8000/liveness/check" -F "video=@video.mp4"

# Head turn only
curl -X POST "http://localhost:8000/liveness/check/head-turn" -F "video=@video.mp4"

# Mouth open only
curl -X POST "http://localhost:8000/liveness/check/mouth" -F "video=@video.mp4"

# Liveness + face match (reference image)
curl -X POST "http://localhost:8000/liveness/compare" \
  -F "video=@liveness_video.mp4" -F "reference=@id_photo.jpg"

# Liveness + face match (reference video)
curl -X POST "http://localhost:8000/liveness/compare" \
  -F "video=@liveness_video.mp4" -F "reference=@reference_video.mp4"
```

## Configuration

Environment variables (optional):

| Variable | Default | Description |
|----------|---------|-------------|
| `liveness_min_valid_frames` | 2 | Minimum frames with detected face |
| `liveness_left_turn_threshold` | 0.50 | Ratio threshold for "turned left" |
| `liveness_mouth_open_threshold` | 0.03 | Min mouth openness (normalized) to count as open |
| `face_match_tolerance` | 0.6 | Face comparison tolerance (lower = stricter) |
| `video_num_frames` | 4 | Frames to extract from video |

## Integration

Use the `/liveness/check` endpoint before approving a transfer:

```python
# 1. User records liveness video
# 2. Call your backend with the video
# 3. Your backend calls this API
response = requests.post(
    "http://localhost:8000/liveness/check",
    files={"video": open("user_video.mp4", "rb")}
)
data = response.json()
if data["status"] == "success" and data["liveness"]["passed"]:
    # Proceed with transfer approval
    pass
else:
    # Reject: liveness failed
    pass
```
