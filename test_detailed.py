"""
Detailed test with artifact detection debugging
"""
import sys
sys.path.insert(0, '.')

from audio_processor import audio_processor
from classifier import voice_classifier
import numpy as np

audio_file = r"audio\sample voice 1.mp3"

print(f"Testing: {audio_file}")
print("="*80)

with open(audio_file, 'rb') as f:
    audio_bytes = f.read()

audio, sr = audio_processor.convert_to_wav(audio_bytes, "mp3")
features_dict = audio_processor.extract_features(audio, sr)
feature_vector = audio_processor.features_to_vector(features_dict)

print(f"Feature vector shape: {feature_vector.shape}")

# Manual artifact check
mfcc_std = feature_vector[13:26]
mfcc_delta_std = feature_vector[39:52]
mfcc_std_avg = np.mean(mfcc_std)
mfcc_delta_std_avg = np.mean(mfcc_delta_std)

print(f"\nManual Artifact Analysis:")
print(f"MFCC Std Avg: {mfcc_std_avg:.4f} (AI if >20)")
print(f"MFCC Delta Std Avg: {mfcc_delta_std_avg:.4f} (AI if <7)")
print(f"Combined AI signature: {mfcc_std_avg > 20 and mfcc_delta_std_avg < 7}")

# Classify
classification, confidence, metadata = voice_classifier.predict(feature_vector)

print(f"\n" + "="*80)
print(f"RESULT: {classification}")
print(f"Confidence: {confidence:.2%}")
print(f"ML AI Probability: {metadata.get('ml_ai_probability', 0):.2%}")
print(f"Artifact Score: {metadata.get('artifact_score', 0):.2%}")
print(f"Detected Artifacts: {metadata['analysis_indicators']['detected_artifacts']}")
print("="*80)

if classification == "AI_GENERATED":
    print("✓ CORRECT")
else:
    print("✗ INCORRECT")
