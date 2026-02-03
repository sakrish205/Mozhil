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
from typing import Optional
import time
import traceback

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
    message: Optional[str] = Field(None, description="Short message describing the test request")
    language: str = Field(..., description="Language of the audio (en, ta, hi, ml, te)")
    audio_format: str = Field(default="mp3", alias="audioFormat", description="Audio format (mp3, wav)")
    audio_base64_format: Optional[str] = Field(None, alias="audioBase64", description="Base64-encoded audio data")
    audio_url: Optional[str] = Field(None, alias="audioUrl", description="URL to the audio file")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "message": "Testing sample voice",
                "language": "en",
                "audioFormat": "mp3",
                "audioBase64": "//uQxAAAAAANIAAAAAExBTUUzLjEwMFVVVVVVVVVVVVVV...",
                "audioUrl": "http://example.com/audio.mp3"
            }
        }


class DetectionResponse(BaseModel):
    """Response model for voice detection."""
    classification: str = Field(..., description="Classification result: 'AI' or 'Human'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    explanation: str = Field(..., description="Detailed explanation of the classification")
    detected_language: Optional[str] = Field(
        default=None,
        description="Detected language of the audio"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "classification": "AI",
                "confidence": 0.87,
                "explanation": "The audio sample exhibits characteristics typical of AI-generated speech...",
                "detected_language": "English"
            }
        }


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
    
    try:
        if request.message:
            print(f"Request message: {request.message}")

        # Get audio data (Base64 or URL)
        audio_bytes = None
        audio_format = request.audio_format.lower() if request.audio_format else "mp3"
        
        if request.audio_base64_format:
            # Process Base64
            try:
                audio_bytes = audio_processor.decode_base64_audio(request.audio_base64_format)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Base64 decoding failed: {str(e)}")
        
        elif request.audio_url:
            # Process URL
            import httpx
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(request.audio_url, timeout=30.0)
                    if resp.status_code != 200:
                        raise HTTPException(status_code=400, detail=f"Failed to download audio from URL. Status: {resp.status_code}")
                    audio_bytes = resp.content
                    
                    # Try to guess format from URL if not specified
                    if not request.audio_format and "." in request.audio_url:
                        ext = request.audio_url.split(".")[-1].lower()
                        if ext in ["mp3", "wav"]:
                            audio_format = ext
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Audio download failed: {str(e)}")
        
        else:
            raise HTTPException(status_code=400, detail="Either audioBase64 or audioUrl is required")

        # Process audio bytes
        try:
            audio, sr = audio_processor.convert_to_wav(audio_bytes, audio_format)
            features = audio_processor.extract_features(audio, sr)
            feature_vector = audio_processor.features_to_vector(features)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio processing error: {str(e)}")
        
        # Classify voice
        classification, confidence, metadata = voice_classifier.predict(feature_vector)
        
        # Get language from request
        target_language = request.language.lower() if request.language else "en"
        if target_language not in ["en", "ta", "hi", "ml", "te"]:
            target_language = "en"
        
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
