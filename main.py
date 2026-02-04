"""
AI Voice Detection API
Multi-language support for Tamil, English, Hindi, Malayalam, and Telugu.

Endpoints:
    POST /api/voice-detection - Analyze audio to determine if AI or human generated
    GET /health - Health check endpoint
"""

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Any
import os
from datetime import datetime
import json
import traceback
import time

# 🔧 Cloud Environment Setup
# Add local 'bin' to PATH so Render/Cloud can find FFmpeg
bin_path = os.path.join(os.getcwd(), "bin")
if os.path.exists(bin_path):
    os.environ["PATH"] = f"{bin_path}{os.pathsep}{os.environ.get('PATH', '')}"
    print(f"Added {bin_path} to PATH for FFmpeg support")

# Import modules
from audio_processor import audio_processor
from classifier import voice_classifier
from explanation_generator import explanation_generator
from language_detector import language_detector

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Detect AI-generated voices in multiple languages (Tamil, English, Hindi, Malayalam, Telugu)",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class DetectionRequest(BaseModel):
    """Request model for voice detection - strictly matches competition format + tester support."""
    language: str = Field(..., description="Language (Tamil, English, Hindi, Malayalam, Telugu)")
    audio_format: str = Field(..., alias="audioFormat", description="Always mp3 for competition")
    audio_base64: Optional[str] = Field(None, alias="audioBase64", description="Base64-encoded MP3 audio")
    audio_url: Optional[str] = Field(None, alias="audioUrl", description="URL to the audio file")
    message: Optional[Any] = Field(None, description="Optional message describing the test request")

    class Config:
        populate_by_name = True


class DetectionResponse(BaseModel):
    """Response model for voice detection - strictly matches competition format."""
    status: str = Field(default="success")
    language: str
    classification: str = Field(..., description="AI_GENERATED or HUMAN")
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class ErrorResponse(BaseModel):
    """Error response model - strictly matches competition format."""
    status: str = Field(default="error")
    message: str


# Helper for consistent error responses
def error_json(message: str, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={"status": "error", "message": message}
    )


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Voice Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoint": "/api/voice-detection"
    }


@app.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "classifier_ready": voice_classifier.is_trained,
        "explanation_system": "Template-based (No Ollama)"
    }


@app.post(
    "/api/voice-detection",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def detect_voice(
    request: DetectionRequest,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key")
):
    """
    Detect whether an audio sample is AI-generated or human-generated.
    Strictly follows the competition format with added tester support for audioUrl.
    """
    # 🛡️ SECURITY: Validate API Key
    VALID_API_KEY = "mozhil-api-key-2024"
    if x_api_key != VALID_API_KEY:
        return error_json("Invalid API key or unauthorized access", 401)

    try:
        # Get audio data (Base64 or URL)
        audio_bytes = None
        
        if request.audio_base64:
            # Process Base64
            try:
                audio_bytes = audio_processor.decode_base64_audio(request.audio_base64)
            except Exception as e:
                return error_json(f"Malformed request: Base64 decoding failed - {str(e)}")
        
        elif request.audio_url:
            # Process URL
            import httpx
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(str(request.audio_url), timeout=30.0)
                    if resp.status_code != 200:
                        return error_json(f"Failed to download audio. Status: {resp.status_code}")
                    audio_bytes = resp.content
            except Exception as e:
                return error_json(f"Audio download failed: {str(e)}")
        
        else:
            return error_json("Either audioBase64 or audioUrl is required")
        
        # Audio format is expected to be mp3 per rules
        audio_format_str = "mp3"
        
        # Process audio bytes
        try:
            audio, sr = audio_processor.convert_to_wav(audio_bytes, audio_format_str)
            features = audio_processor.extract_features(audio, sr)
            feature_vector = audio_processor.features_to_vector(features)
        except Exception as e:
            return error_json(f"Audio processing error: {str(e)}")
        
        # Classify voice
        classification, confidence, metadata = voice_classifier.predict(feature_vector)
        
        # Target language from request (Tamil, English, etc.)
        target_language = request.language
        
        # Generate explanation
        explanation = explanation_generator.generate_explanation(
            classification=classification,
            confidence=confidence,
            metadata=metadata,
            target_language=target_language
        )
        
        return DetectionResponse(
            status="success",
            language=target_language,
            classification=classification,
            confidenceScore=round(confidence, 4),
            explanation=explanation
        )
        
    except Exception as e:
        traceback.print_exc()
        return error_json(f"Internal server error: {str(e)}", 500)


# Error handlers to match competition format
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return error_json(f"Malformed request: {str(exc.errors())}", 422)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return error_json(str(exc.detail), exc.status_code)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return error_json(f"Internal server error: {str(exc)}", 500)


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
