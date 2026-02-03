# 🎓 Beginner's Guide: AI Voice Detection

Welcome! This guide will take you from a fresh installation to successfully testing your own voice recordings.

---

## 🛠️ Step 1: Install PREREQUISITES
Before running the code, you need three main tools installed on your system:

### 1. Python 3.9+
Download and install from [python.org](https://www.python.org/). **Make sure to check "Add Python to PATH" during installation.**

### 2. FFmpeg (Audio Processor)
This is a background tool that helps the program "hear" your audio.
*   **Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/). Extract and add the `bin` folder to your System PATH.
*   Check it by typing `ffmpeg -version` in a terminal.

---

## 🐍 Step 2: Setup the Project
1.  Open a terminal in the project folder (`Mozhil`).
2.  Install the Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Step 3: Run the API Server
The server must be running to process requests.
```bash
python main.py
```
Wait until you see: `Uvicorn running on http://0.0.0.0:8000`. **Keep this window open!**

---

## 🌐 Step 4: Make it Public (Ngrok)
To use the competition website, your server needs a public address.
1.  Download [ngrok](https://ngrok.com/).
2.  In a **new** terminal window, run:
    ```bash
    ngrok http 8000
    ```
3.  Copy the `Forwarding` URL (looks like `https://abc-123.ngrok-free.dev`).

---

## 🎤 Step 5: Convert Your Audio
The API doesn't accept `.wav` files directly; it needs a "Base64" text version.
1.  Put your audio file (e.g., `my_voice.wav`) in the `audio/` folder.
2.  Run the converter:
    ```bash
    python convert_audio.py
    ```
3.  Enter the path when asked: `audio/my_voice.wav`.
4.  Open the newly created `Audio Base64 Format.txt` and **copy all the text inside**.

---

## 🧪 Step 6: Testing the Result
Go to your testing tool or competition form and fill it as follows:

- **Endpoint URL:** `https://[YOUR-NGROK-URL]/api/voice-detection`
- **x-api-key:** `mozhil-api-key-2024`
- **Language:** `Tamil`, `English`, `Hindi`, `Malayalam`, or `Telugu`
- **Audio Format:** `mp3`
- **Audio Base64 Format:** *Paste the text you copied in Step 5.*
- **Audio URL:** (Optional) Use a direct link to an MP3 instead of Base64.

### 🎯 Evaluation Tip
The competition uses strict classification values:
- `AI_GENERATED`
- `HUMAN`

Our API is fully aligned with these rules.

---

## ❓ Troubleshooting
### 1. "404 Not Found"
*   **Cause:** Your URL is wrong or the server isn't running.
*   **Fix:** Ensure the URL ends in `/detect`. Check if `python main.py` is still running.

### 2. "422 Unprocessable Entity"
*   **Cause:** Typo in field names (like `language` or `audioBase64`).
*   **Fix:** Ensure names are exactly as shown above (case-sensitive).

### 3. "500 Internal Server Error"
*   **Cause:** FFmpeg might not be installed correctly or there's a file permissions issue.
*   **Fix:** Run `ffmpeg -version` to verify it works. Check the terminal for error logs.
