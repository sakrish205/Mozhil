"""
Analyze the actual feature characteristics of the AI sample
"""
import sys
sys.path.insert(0, '.')

from audio_processor import audio_processor
import numpy as np

# Test the sample AI-generated voice
audio_file = r"audio\sample voice 1.mp3"

print(f"Analyzing: {audio_file}")
print("="*80)

# Read and process
with open(audio_file, 'rb') as f:
    audio_bytes = f.read()

audio, sr = audio_processor.convert_to_wav(audio_bytes, "mp3")
features = audio_processor.extract_features(audio, sr)
feature_vector = audio_processor.features_to_vector(features)

print(f"\nFeature Vector Analysis:")
print(f"Total features: {len(feature_vector)}")
print(f"Mean: {np.mean(feature_vector):.4f}")
print(f"Std: {np.std(feature_vector):.4f}")
print(f"Min: {np.min(feature_vector):.4f}")
print(f"Max: {np.max(feature_vector):.4f}")

# Analyze key regions
mfcc_mean = feature_vector[:13]
mfcc_std = feature_vector[13:26]
mfcc_delta_mean = feature_vector[26:39]
mfcc_delta_std = feature_vector[39:52]

print(f"\nMFCC Statistics:")
print(f"MFCC Mean - Avg: {np.mean(mfcc_mean):.4f}, Std: {np.std(mfcc_mean):.4f}")
print(f"MFCC Std - Avg: {np.mean(mfcc_std):.4f}, Std: {np.std(mfcc_std):.4f}")
print(f"MFCC Delta Mean - Avg: {np.mean(mfcc_delta_mean):.4f}")
print(f"MFCC Delta Std - Avg: {np.mean(mfcc_delta_std):.4f}")

# Check for periodicity
fft_features = np.fft.fft(feature_vector[:50])
spectral_peaks = np.abs(fft_features)
print(f"\nPeriodicity Analysis:")
print(f"Max FFT peak: {np.max(spectral_peaks):.4f}")
print(f"Median FFT: {np.median(spectral_peaks):.4f}")
print(f"Peak-to-Median ratio: {np.max(spectral_peaks) / (np.median(spectral_peaks) + 1e-10):.2f}")

print("\n" + "="*80)
print("KEY INSIGHT: These values show the 'signature' of this AI voice.")
print("We need to train the model to recognize THESE specific patterns.")
