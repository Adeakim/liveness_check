from pydantic import BaseModel


class LivenessResult(BaseModel):
    passed: bool
    message: str
    details: dict | None = None


class LivenessResponse(BaseModel):
    status: str  # "success" | "failed" | "error"
    liveness: LivenessResult | None = None
    error: str | None = None


class FaceMatchResult(BaseModel):
    matched: bool
    distance: float
    message: str


class CompareResponse(BaseModel):
    status: str  # "success" | "failed" | "error"
    passed: bool  # True only if liveness passed AND face matched
    liveness: LivenessResult | None = None
    face_match: FaceMatchResult | None = None
    error: str | None = None
