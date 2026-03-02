from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import liveness


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload MediaPipe model
    yield
    # Shutdown


app = FastAPI(
    title="Liveness Check API",
    description="API for verifying liveness via head movement detection. Use before approving transfers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(liveness.router)


@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}
