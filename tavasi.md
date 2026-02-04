# Mozhil: Advanced AI Voice Authenticity & Linguistic Analysis System

## 1. Executive Summary
**Mozhil** is a high-performance audio analysis framework designed to verify the provenance of vocal signals. It utilizes **Digital Signal Processing (DSP)** and **Hybrid Machine Learning** to differentiate between authentic human speech and synthetic (AI-generated) voice clones across five major languages: Tamil, English, Hindi, Malayalam, and Telugu.

## 2. Technical Architecture
The system is built on a modular microservice architecture using **FastAPI** for high-concurrency handling.
- **Ingestion Layer**: Supports asynchronous processing of Base64-encoded streams and remote URI fetching.
- **Normalization Engine**: Standardizes signals to a 16kHz Mono-PCM format to eliminate variable-bitrate artifacts.
- **Extraction Core**: Analyzes the acoustic "DNA" through 106-dimensional feature vectors.
- **Inference Layer**: Utilizes a Hybrid detection system combining Random Forest ML with rule-based artifact detection.
- **Localized XAI**: Implements **Explainable AI** to provide human-readable justification for every classification in the target language.

## 3. Signal Analysis & Feature Extraction
The core of Mozhil's detection capability lies in its deep analysis of acoustic signatures:
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Captures the physical characteristics of the human vocal tract.
- **Spectral Centroid & Bandwidth**: Analyzes the "brightness" and frequency spread to detect digital filtering patterns.
- **Zero Crossing Rate (ZCR)**: Identifies high-frequency noise profiles typical of neural vocoder synthesis.
- **RMS Energy Analysis**: Monitors amplitude consistency, as AI voices often lack the natural volume fluctuations (shimmer) of human breath.

## 4. AI Detection Methodology - The Modern AI Voice Paradox
The system identifies synthetic voices through a **Hybrid Detection Architecture** combining Machine Learning with rule-based acoustic analysis.

### The Key Innovation
Traditional AI detection looks for "robotic" artifacts. However, modern neural TTS engines (like ElevenLabs, Google WaveNet) are sophisticated enough to sound natural. Mozhil detects them through a critical insight:

**Modern AI voices have high spectral variance (natural sound) BUT low delta variance (artificial smoothness).**

This means:
- **MFCC Variance**: High (~30) - sounds natural to the human ear
- **MFCC Delta Variance**: Low (~5.5) - transitions between phonemes are mathematically too smooth

Real human speech has micro-jitters, breath variations, and irregular timing. AI models, despite their sophistication, produce unnaturally consistent transitions.

### Detection Pipeline
1. **Machine Learning Layer**: Random Forest classifier trained on acoustic feature patterns
2. **Rule-Based Artifact Detection**: Analyzes specific signatures:
   - **Temporal Micro-jitters**: Measures consistency in MFCC deltas
   - **Formant Transition Analysis**: Detects overly smooth vowel transitions
   - **Spectral Continuity**: Identifies periodic artifacts from neural vocoders
3. **Weighted Decision**: Combines both approaches (30% ML, 70% Artifacts) for final classification

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
- **Intelligence Core**: Scikit-Learn (Random Forest + Custom Rule Engine)
- **Audio Processing**: FFmpeg & Pydub
- **Containerization**: Docker (Multi-stage build for lightweight deployment)

## 8. Novelty & Competitive Advantages

### What Makes Mozhil Unique?

1. **Multi-Lingual Explainable AI (XAI)**
   - Unlike competitors that just output "AI" or "Human," Mozhil provides detailed explanations in the user's native language (Tamil, Telugu, Malayalam, Hindi, English)
   - Critical for legal and forensic applications where decisions must be justified

2. **Indic Language Specialization**
   - First system with custom phonetic profiles for Tamil, Telugu, Malayalam, and Hindi
   - Western tools (Resemble AI, Deepware) only work well with English

3. **Privacy-First Architecture**
   - 100% on-premise processing - no data sent to external APIs
   - Essential for banks, government agencies, and legal institutions

4. **Real-World Robustness**
   - Trained on noisy, compressed audio (WhatsApp calls, phone recordings)
   - Most research models fail in real-world conditions

5. **Hybrid Detection**
   - Combines Signal Processing with Machine Learning
   - Catches both obvious AI voices and sophisticated deepfakes

## 9. Use Cases

### Where Can This Be Deployed?

1. **Banking & Financial Fraud Prevention**: Verify customer identity during phone banking
2. **Legal & Forensic Evidence**: Courts can verify if audio evidence is authentic
3. **Media & Journalism Verification**: News agencies can verify leaked audio clips
4. **Call Centers & Customer Support**: Detect AI voice-changing software
5. **Social Media Platforms**: Flag AI-generated voice content to prevent scams
6. **Government & National Security**: Verify intercepted communications
