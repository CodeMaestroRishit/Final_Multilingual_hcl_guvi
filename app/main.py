from dotenv import load_dotenv
import os

# Load env before importing other modules
load_dotenv()

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router as api_router
from app.models.detector import VoiceDetector
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request, HTTPException

# Lifecycle manager to preload model
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    print("Initializing application...")
    VoiceDetector.get_instance()
    yield
    print("Shutting down...")

app = FastAPI(
    title="AI Voice Detector API",
    description="Hackathon API for detecting AI-generated speech using Wav2Vec2",
    version="1.0.0",
    lifespan=lifespan
)

# --- Fix for 405 Method Not Allowed & CORS ---
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins for hackathon testing
    allow_credentials=True,
    allow_methods=["*"],  # Allow ALL methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allow ALL headers (x-api-key, content-type, etc.)
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Prompt example: "Invalid API key or malformed request"
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Invalid API key or malformed request"},
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/ping")
def ping():
    return {"status": "awake", "message": "Server is warm and ready"}

app.include_router(api_router, prefix="")

@app.get("/")
async def root():
    return {"message": "AI Voice Detector API", "endpoints": ["/health", "/detect"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
