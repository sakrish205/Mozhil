# 🎓 Mozhil: AI Voice Detection User Guide

Welcome! This guide will take you from a fresh installation to successfully testing your own voice recordings with the **Mozhil** system.

---

## 🛠️ Step 1: Install PREREQUISITES
Before running the code, you need three main tools installed on your system:

### 1. Python 3.9+
Download and install from [python.org](https://www.python.org/). **Make sure to check "Add Python to PATH" during installation.**

### 2. FFmpeg (Audio Processor)
This is a background tool that helps the program "hear" and decode your audio.
*   **Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/). Extract and add the `bin` folder to your System PATH.
*   **Verify:** Type `ffmpeg -version` in a terminal. If it shows a version, you're ready!

---

## 🐍 Step 2: Setup the Project
1.  Open your terminal inside the `Mozhil` project directory.
2.  Install all required AI and Audio libraries:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Step 3: Run the API Server
The server must be active to process requests. Start it with:
```bash
python main.py
```
Wait until you see: `Uvicorn running on http://0.0.0.0:8000`. **Keep this terminal window open!**

---

## 🌐 Step 4: Accessing the API
### Local Testing
Your API is available locally at: `http://localhost:8000/api/voice-detection`

### Public Access (Ngrok)
If you need to test from an external website or competition portal:
1.  Open a **new** terminal and run:
    ```bash
    ngrok http 8000
    ```
2.  Use the `Forwarding` URL (e.g., `https://xyz-123.ngrok-free.dev`) as your Base URL.

---

## 🎤 Step 5: Preparing Your Audio (Base64)
The API requires audio to be sent as a **Base64 string**.
1.  Place your MP3 or WAV file in the `audio/` folder inside the project.
2.  Run the conversion script:
    ```bash
    python convert_audio.py
    ```
3.  Follow the prompts to select your file. It will create a file named `Audio Base64 Format.txt`.
4.  Copy the entire content of that text file.

---

## 🧪 Step 6: Testing the Endpoint
Use a tool like Postman, Thunder Client, or the competition tester:

- **Method:** `POST`
- **URL:** `[YOUR-SERVER-URL]/api/voice-detection`
- **Header:** `x-api-key: mozhil-api-key-2026`
- **Body (JSON):**
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "PASTE_YOUR_BASE64_STRING_HERE"
}
```

---

## 🎯 Understanding the Results
Our system provides more than just a label. You will receive:
1.  **Classification**: Either `AI_GENERATED` or `HUMAN`.
2.  **Confidence Score**: A value from 0.0 to 1.0 (e.g., 0.95 means 95% sure).
3.  **Detailed Explanation**: A multi-lingual reasoning (in Tamil, English, etc.) explaining **why** the AI reached that conclusion (e.g., "Detected unnatural spectral consistency").

---

## ❓ Troubleshooting
### 1. "401 Unauthorized"
*   **Fix:** Check your `x-api-key`. It must be exactly `mozhil-api-key-2026`.

### 2. "422 Unprocessable Entity"
*   **Cause:** Typo in JSON field names.
*   **Fix:** Ensure you use `language`, `audioFormat`, and `audioBase64` exactly as shown (case-sensitive).

### 3. "500 Internal Server Error"
*   **Cause:** Usually means FFmpeg is missing or the audio data is corrupted.
*   **Fix:** Ensure FFmpeg is in your system PATH. Try a standard MP3 file.

### 4. Memory Errors
*   **Note:** Mozhil is optimized for 512MB RAM environments. If you get memory errors, ensure you aren't trying to process audio longer than 30 seconds.
