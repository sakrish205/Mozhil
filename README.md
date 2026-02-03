# 🎙️ Mozhil - AI Voice Detection API

Mozhil is a high-performance, multi-language AI-generated voice detection system. It analyzes audio samples to distinguish between human speech and AI-cloned voices using advanced spectral analysis and machine learning.

---

## 🚦 Getting Started
If you are new to the project, please start with our **[Beginner Tutorial](file:///c:/Users/Saketha/Desktop/projects/Mozhil/tutorial.md)**. It contains a step-by-step guide for installation, testing, and troubleshooting.

### Quick Commands
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull the AI explanation model
ollama pull translategemma:4b

# 3. Start the detection server
python main.py
```

---

## 🎯 Competition Tester Info
When submitting or testing via the competition portal, use the following settings:

| Field | Value |
|-------|-------|
| **x-api-key** | `mozhil-api-key-2024` |
| **Endpoint URL** | `https://YOUR-NGROK-URL.ngrok-free.dev/detect` |
| **Language** | `en`, `ta`, `hi`, `ml`, or `te` |
| **Audio Format** | `wav` (recommended) or `mp3` |
| **Audio Base64 Format** | Contents of `Audio Base64 Format.txt` |

### Supported Languages
- 🇬🇧 `en` - English
- 🇮🇳 `ta` - Tamil (தமிழ்)
- 🇮🇳 `hi` - Hindi (हिंदी)
- 🇮🇳 `ml` - Malayalam (മലയാളം)
- 🇮🇳 `te` - Telugu (తెలుగు)

---

## 📡 API Reference

### `POST /detect`
The primary endpoint for voice analysis.

**Headers:**
```http
x-api-key: mozhil-api-key-2024
Content-Type: application/json
```

**Request Body:**
```json
{
  "language": "en",
  "audioFormat": "wav",
  "audioBase64": "UklGRuS6AgBXQVZF..."
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
- `explanation_generator.py`: Uses Ollama to explain *why* a voice was flagged.

---

## ⚙️ Requirements
- **Python 3.9+**
- **FFmpeg**: Required for audio processing.
- **Ollama**: Required for generating natural language explanations.
