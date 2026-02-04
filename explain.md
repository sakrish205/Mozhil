# Mozhil: Advanced AI Voice Authenticity & Linguistic Analysis System

## 1. Executive Summary
**Mozhil** is a high-performance audio analysis framework designed to verify the provenance of vocal signals. It utilizes **Digital Signal Processing (DSP)** and **Exsemble Machine Learning** to differentiate between authentic human speech and synthetic (AI-generated) voice clones across five major languages: Tamil, English, Hindi, Malayalam, and Telugu.

## 2. Technical Architecture
The system is built on a modular microservice architecture using **FastAPI** for high-concurrency handling.
- **Ingestion Layer**: Supports asynchronous processing of Base64-encoded streams and remote URI fetching.
- **Normalization Engine**: Standardizes signals to a 16kHz Mono-PCM format to eliminate variable-bitrate artifacts.
- **Extraction Core**: Analyzes the acoustic "DNA" through 106-dimensional feature vectors.
- **Inference Layer**: Utilizes a Random Forest Ensemble model calibrated for spectral-temporal artifact detection.
- **Localized XAI**: Implements **Explainable AI** to provide human-readable justification for every classification in the target language.

## 3. Signal Analysis & Feature Extraction
The core of Mozhil's detection capability lies in its deep analysis of acoustic signatures:
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Captures the physical characteristics of the human vocal tract.
- **Spectral Centroid & Bandwidth**: Analyzes the "brightness" and frequency spread to detect digital filtering patterns.
- **Zero Crossing Rate (ZCR)**: Identifies high-frequency noise profiles typical of neural vocoder synthesis.
- **RMS Energy Analysis**: Monitors amplitude consistency, as AI voices often lack the natural volume fluctuations (shimmer) of human breath.

## 4. AI Detection Methodology (The "Perfect" Artifact)
The system identifies synthetic voices by detecting what is known as **"Artificial Perfection"**:
- **Temporal Micro-jitters**: Human speech contains tiny, irregular timing errors. AI models are often mathematically too precise.
- **Formant Transition Analysis**: Mozhil analyzes the fluid transition between phonemes. In synthetic speech, these transitions often follow rigid mathematical curves rather than organic muscle movements.
- **Spectral Continuity**: Neural synthesis engines (like WaveNet) often leave subtle periodic artifacts in the spectrogram that the Ensemble model is trained to flag.

## 5. Multi-lingual Acoustic Profiling
Language detection and analysis are performed through **Phonetic Energy Mapping**:
- **Dravidian Language Profiling**: Tamil and Telugu are identified by specific energy clusters in the mid-high frequency range, associated with retroflex consonants (e.g., 'ழ', 'ట').
- **Phonetic Frequency Baselines**: Each supported language (Tamil, English, Hindi, Malayalam, Telugu) has a unique "Spectral Signature." The system compares the incoming signal's bandwidth and spectral rolloff against these optimized language profiles to determine the linguistic context.

## 6. Data Integrity & Training Sources
To ensure zero-bias and high accuracy, the underlying intelligence is built upon a **Hybrid Multi-Dataset Framework**:
- **Human Baseline**: Sourced from **Mozilla Common Voice** and **AI4Bharat (Bhashini)**, providing clean, diverse vocal samples across age, gender, and regional accents.
- **Synthetic Profiles**: Generated using state-of-the-art architectures, including Neural TTS (Google/Amazon), WaveGlow, and high-fidelity clones from ElevenLabs.
- **Research Standard**: Integrated insights from the **ASVspoof Global Challenge** (the industry gold standard for voice anti-spoofing research).
- **Environmental Robustness**: Augmented with real-world noise (Gaussian noise, office hum, and GSM-codec compression) to ensure accuracy in non-studio conditions.

## 7. Technology Stack (STC)
- **Framework**: FastAPI (Asynchronous Python)
- **DSP Libraries**: Librosa, SoundFile, NumPy, SciPy
- **Intelligence Core**: Scikit-Learn (Random Forest Ensemble)
- **Audio Processing**: FFmpeg & Pydub
- **Containerization**: Docker (Multi-stage build for lightweight deployment)
