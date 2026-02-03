# 🎙️ Mozhil - AI Voice Detection API

Mozhil is a high-performance, multi-language AI-generated voice detection system. It analyzes audio samples to distinguish between human speech and AI-cloned voices using advanced spectral analysis and machine learning.

---

## 🚦 Getting Started
If you are new to the project, please start with our **[Beginner Tutorial](file:///c:/Users/Saketha/Desktop/projects/Mozhil/tutorial.md)**. It contains a step-by-step guide for installation, testing, and troubleshooting.

### Quick Commands
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the detection server
python main.py
```

---

## 🎯 Competition Tester Info
When submitting or testing via the competition portal, use the following settings:

| Field | Value |
|-------|-------|
| **x-api-key** | `mozhil-api-key-2024` |
| **Endpoint URL** | `https://YOUR-NGROK-URL.ngrok-free.dev/api/voice-detection` |
| **Language** | `Tamil`, `English`, `Hindi`, `Malayalam`, or `Telugu` |
| **Audio Format** | `mp3` |
| **Audio Base64 Format** | Contents of `Audio Base64 Format.txt` |

---

## 📡 API Reference

### `POST /api/voice-detection`
The primary endpoint for voice analysis.

**Headers:**
```http
x-api-key: mozhil-api-key-2024
Content-Type: application/json
```

**Request Body:**
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0...",
  "audioUrl": "https://example.com/audio.mp3",
  "message": "Tester verification"
}
```
*Note: Use either `audioBase64` or `audioUrl`. `message` is optional.*

**Response Body (Success):**
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

---

## 🌐 Public Access (Ngrok)
To allow the competition tester to reach your local server, use ngrok:
```bash
ngrok http 8000
```
*Note: Every time you restart ngrok, the "Endpoint URL" will change. Update it in your tester accordingly.*

---

## 📁 Project Structure
- `main.py`: The API server entry point.
- `audio_processor.py`: handles feature extraction (MFCCs).
- `classifier.py`: The core machine learning logic.
- `convert_audio.py`: Helper script to generate Base64 strings.
- `explanation_generator.py`: Generates localized explanations using predefined templates.

---

## ⚙️ Requirements
- **Python 3.9+**
- **FFmpeg**: Required for audio processing.
