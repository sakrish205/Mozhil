"""
Test script for the AI Voice Detection API.
Creates a simple test audio and sends it to the API.
"""

import base64
import json
import requests
import sys
import os
import tempfile
import numpy as np

try:
    from scipy.io import wavfile
except ImportError:
    print("Installing scipy...")
    os.system("pip install scipy")
    from scipy.io import wavfile


def create_test_audio() -> str:
    """Create a simple test audio file and return base64 encoded data."""
    # Generate a simple sine wave (simulating speech-like audio)
    sample_rate = 16000
    duration = 2.0  # seconds
    frequency = 440  # Hz (A4 note)
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a more complex waveform (mix of frequencies like speech)
    audio = np.sin(2 * np.pi * 200 * t) * 0.3  # Fundamental
    audio += np.sin(2 * np.pi * 400 * t) * 0.2  # Harmonic
    audio += np.sin(2 * np.pi * 800 * t) * 0.1  # Higher harmonic
    
    # Add some modulation (like natural speech)
    audio *= (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save to temp file
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(temp_wav.name, sample_rate, audio_int16)
    temp_wav.close()
    
    # Try to convert to MP3 if pydub/ffmpeg available
    try:
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_wav(temp_wav.name)
        temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        audio_segment.export(temp_mp3.name, format="mp3")
        temp_mp3.close()
        
        with open(temp_mp3.name, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        os.unlink(temp_mp3.name)
        os.unlink(temp_wav.name)
        return audio_base64
        
    except Exception as e:
        print(f"Note: Could not convert to MP3 ({e}), using WAV")
        with open(temp_wav.name, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        os.unlink(temp_wav.name)
        return audio_base64


def test_api(api_url: str = "http://localhost:8000", api_key: str = None):
    """Test the voice detection API."""
    
    print("=" * 60)
    print("AI Voice Detection API Test")
    print("=" * 60)
    
    # Test health endpoint
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {json.dumps(response.json(), indent=4)}")
        else:
            print(f"   Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print(f"   ERROR: Could not connect to {api_url}")
        print("   Make sure the API server is running:")
        print("   python -m uvicorn main:app --host 0.0.0.0 --port 8000")
        return False
    
    # Test detect endpoint
    print("\n2. Creating test audio...")
    try:
        audio_base64 = create_test_audio()
        print(f"   Audio created: {len(audio_base64)} characters (base64)")
    except Exception as e:
        print(f"   ERROR creating test audio: {e}")
        return False
    
    print("\n3. Testing /detect endpoint...")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    
    payload = {
        "language": "en",
        "audio_format": "mp3",
        "audio_base64_format": audio_base64
    }
    
    try:
        response = requests.post(
            f"{api_url}/detect",
            json=payload,
            headers=headers,
            timeout=60
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n   Classification: {result.get('classification')}")
            print(f"   Confidence: {result.get('confidence'):.2%}")
            print(f"   Detected Language: {result.get('detected_language')}")
            print(f"\n   Explanation:")
            print(f"   {result.get('explanation')}")
            print("\n" + "=" * 60)
            print("TEST PASSED!")
            print("=" * 60)
            return True
        else:
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ERROR: Request timed out")
        return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the AI Voice Detection API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authorization header"
    )
    
    args = parser.parse_args()
    
    success = test_api(args.url, args.api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
