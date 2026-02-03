"""
AI Voice Detection API
Multi-language support for Tamil, English, Hindi, Malayalam, and Telugu.

Endpoints:
    POST /detect - Analyze audio to determine if AI or human generated
    GET /health - Health check endpoint
"""

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Any
import time
import traceback
import json
from datetime import datetime
import os

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
    """Request model for voice detection - matches competition format."""
    message: Optional[Any] = Field(None, description="Short message or data describing the test request")
    language: Optional[Any] = Field(default="en", description="Language of the audio (en, ta, hi, ml, te)")
    audio_format: Optional[Any] = Field(default="mp3", alias="audioFormat", description="Audio format (mp3, wav)")
    audio_base64_format: Optional[Any] = Field(None, alias="audioBase64", description="Base64-encoded audio data")
    audio_url: Optional[Any] = Field(None, alias="audioUrl", description="URL to the audio file")

    class Config:
        populate_by_name = True


class DetectionResponse(BaseModel):
    """Response model for voice detection."""
    classification: str = Field(..., description="Classification result: 'AI' or 'Human'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    explanation: str = Field(..., description="Detailed explanation of the classification")
    detected_language: Optional[str] = Field(
        default=None,
        description="Detected language of the audio"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    classifier_ready: bool
    ollama_available: bool


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None


# Helper: Log Interaction for Honeypot analysis
def log_interaction(event_type, data, level="info"):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "level": level,
        "data": data
    }
    
    # Print to console
    color = "\033[92m" if level == "info" else "\033[91m"
    print(f"{color}[{event_type}] {json.dumps(data)}\033[0m")
    
    # Save to file
    log_file = "interactions.jsonl"
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error writing to log: {e}")


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Voice Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "detect": "POST /detect",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        classifier_ready=voice_classifier.is_trained,
        ollama_available=explanation_generator.available_model is not None
    )


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def detect_voice(
    request: DetectionRequest,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key")
):
    """
    Detect whether an audio sample is AI-generated or human-generated.
    """
    start_time = time.time()
    
    # 🛡️ SECURITY: Validate API Key (Required for Honeypot Tester)
    VALID_API_KEY = "mozhil-api-key-2024"
    if x_api_key != VALID_API_KEY:
        # Log unauthenticated attempt
        log_interaction("UNAUTHORIZED_ATTEMPT", {"provided_key": x_api_key}, "error")
        raise HTTPException(
            status_code=401, 
            detail="Unauthorized: Invalid or missing API key"
        )

    # 📝 HONEPOT LOGGING: Record the interaction
    log_interaction("DETECT_REQUEST", {
        "language": request.language,
        "format": request.audio_format,
        "has_base64": bool(request.audio_base64_format),
        "has_url": bool(request.audio_url),
        "message": request.message
    })
    
    try:
        # Get audio data (Base64 or URL)
        audio_bytes = None
        audio_format_str = str(request.audio_format).lower() if request.audio_format else "mp3"
        
        if request.audio_base64_format:
            # Process Base64
            try:
                audio_bytes = audio_processor.decode_base64_audio(str(request.audio_base64_format))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Base64 decoding failed: {str(e)}")
        
        elif request.audio_url:
            # Process URL
            import httpx
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(str(request.audio_url), timeout=30.0)
                    if resp.status_code != 200:
                        raise HTTPException(status_code=400, detail=f"Failed to download audio. Status: {resp.status_code}")
                    audio_bytes = resp.content
                    
                    # Try to guess format from URL if not specified
                    if not request.audio_format and "." in str(request.audio_url):
                        ext = str(request.audio_url).split(".")[-1].lower()
                        if ext in ["mp3", "wav"]:
                            audio_format_str = ext
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Audio download failed: {str(e)}")
        
        else:
            # 🛡️ HONEPOT FEATURE: If no audio but message exists, it's a "Probe"
            if request.message:
                log_interaction("PROBE_DETECTED", {"message": request.message})
                return DetectionResponse(
                    classification="AI",
                    confidence=0.99,
                    explanation="Automated agentic probe detected. This interaction has been logged for security analysis as part of our honeypot protocol.",
                    detected_language="Detecting..."
                )
            raise HTTPException(status_code=400, detail="Either audioBase64 or audioUrl is required")

        # Process audio bytes
        try:
            audio, sr = audio_processor.convert_to_wav(audio_bytes, audio_format_str)
            features = audio_processor.extract_features(audio, sr)
            feature_vector = audio_processor.features_to_vector(features)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")
        
        # Classify voice
        classification, confidence, metadata = voice_classifier.predict(feature_vector)
        
        # Get language from request
        lang_input = str(request.language).lower() if request.language else "en"
        target_language = lang_input if lang_input in ["en", "ta", "hi", "ml", "te"] else "en"
        
        lang_name = {
            "en": "English",
            "ta": "Tamil", 
            "hi": "Hindi",
            "ml": "Malayalam",
            "te": "Telugu"
        }.get(target_language, "English")
        
        # Generate explanation
        explanation = explanation_generator.generate_explanation(
            classification=classification,
            confidence=confidence,
            metadata=metadata,
            target_language=target_language
        )
        
        # Log processing time
        process_time = time.time() - start_time
        print(f"Request processed in {process_time:.2f}s")
        
        return DetectionResponse(
            classification=classification,
            confidence=round(confidence, 4),
            explanation=explanation,
            detected_language=lang_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# Error handlers
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the invalid interaction
    try:
        body_bytes = await request.body()
        body_str = body_bytes.decode() if body_bytes else None
    except:
        body_str = "could not decode body"

    log_interaction("INVALID_REQUEST_BODY", {
        "errors": exc.errors(),
        "body": body_str
    }, "warning")
    
    return JSONResponse(
        status_code=422,
        content={"error": "Validation failed", "detail": exc.errors()}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "detail": None}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
