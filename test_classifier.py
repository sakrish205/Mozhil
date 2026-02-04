"""
Quick test script to verify AI voice detection on sample audio
"""
import sys
sys.path.insert(0, '.')

from audio_processor import audio_processor
from classifier import voice_classifier

# Test the sample AI-generated voice
audio_file = r"audio\sample voice 1.mp3"

print(f"Testing: {audio_file}")
print("="*60)

# Read the file
with open(audio_file, 'rb') as f:
    audio_bytes = f.read()

# Process
print("Converting to WAV...")
audio, sr = audio_processor.convert_to_wav(audio_bytes, "mp3")

print("Extracting features...")
features = audio_processor.extract_features(audio, sr)

print("Converting to vector...")
feature_vector = audio_processor.features_to_vector(features)

print(f"Feature vector shape: {feature_vector.shape}")

# Classify
print("\nRunning classification...")
classification, confidence, metadata = voice_classifier.predict(feature_vector)

print("="*60)
print(f"RESULT: {classification}")
print(f"Confidence: {confidence:.2%}")
print(f"Human Probability: {metadata['human_probability']:.2%}")
print(f"AI Probability: {metadata['ai_probability']:.2%}")
print("="*60)

if classification == "AI_GENERATED":
    print("✓ CORRECT: Detected as AI-generated")
else:
    print("✗ INCORRECT: Should be AI-generated but detected as HUMAN")
    print("\nThis means the model needs further tuning.")
