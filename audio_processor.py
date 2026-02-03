"""
Audio Processor Module
Handles decoding, conversion, and feature extraction from audio files.
"""

import base64
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError:
    raise ImportError("Please install librosa and soundfile: pip install librosa soundfile")

try:
    from pydub import AudioSegment
except ImportError:
    raise ImportError("Please install pydub: pip install pydub")


class AudioProcessor:
    """Process audio files for voice classification."""
    
    # Target sample rate for processing
    SAMPLE_RATE = 16000
    
    # MFCC parameters
    N_MFCC = 13
    N_FFT = 2048
    HOP_LENGTH = 512
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
    
    def decode_base64_audio(self, base64_audio: str) -> bytes:
        """Decode base64 encoded audio data."""
        if not base64_audio:
            raise ValueError("Audio data is empty")
            
        # Check if user pasted the command instead of the result
        if "[Convert]::" in base64_audio or "ReadAllBytes" in base64_audio:
            raise ValueError("You pasted the PowerShell command instead of the audio data. Please run the command in your terminal first, then copy the result (the long string of characters) and paste it here.")
            
        if base64_audio.startswith("C:\\") or base64_audio.startswith("D:\\"):
             raise ValueError("You pasted a file path instead of the audio data. Please use an online Base64 converter or the PowerShell command to convert the file to text first.")

        try:
            # Remove data URL prefix if present
            if "," in base64_audio:
                base64_audio = base64_audio.split(",")[1]
            
            # Remove all whitespace, quotes, and common non-base64 characters
            base64_audio = "".join(c for c in base64_audio if c.isalnum() or c in "+/=")
            
            # Add padding if missing
            padding = len(base64_audio) % 4
            if padding:
                base64_audio += "=" * (4 - padding)
            
            return base64.b64decode(base64_audio)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio: {str(e)}")
    
    def convert_to_wav(self, audio_bytes: bytes, input_format: str = "mp3") -> Tuple[np.ndarray, int]:
        """
        Convert audio bytes to WAV format and load as numpy array.
        """
        # Create temporary files
        temp_input = os.path.join(self.temp_dir, f"input_audio.{input_format}")
        temp_output = os.path.join(self.temp_dir, f"output_audio_{os.getpid()}.wav")
        
        try:
            # Write input audio to temp file
            with open(temp_input, "wb") as f:
                f.write(audio_bytes)
            
            # Try to load with specified format first
            try:
                audio = AudioSegment.from_file(temp_input, format=input_format)
            except Exception as e:
                print(f"Failed to load as {input_format}, trying auto-detect: {e}")
                # Fallback to auto-detect
                audio = AudioSegment.from_file(temp_input)
            
            # Convert to mono and set sample rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(self.SAMPLE_RATE)
            
            # Export as WAV
            audio.export(temp_output, format="wav")
            
            # Load with librosa
            y, sr = librosa.load(temp_output, sr=self.SAMPLE_RATE, mono=True)
            
            return y, sr
            
        finally:
            # Cleanup temp files
            for f in [temp_input, temp_output]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract audio features for classification.
        
        Features extracted:
        - MFCCs (13 coefficients + deltas + delta-deltas)
        - Spectral centroid
        - Spectral rolloff
        - Zero crossing rate
        - RMS energy
        - Spectral contrast
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=self.N_MFCC,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH
        )
        
        # MFCC deltas (velocity)
        mfcc_delta = librosa.feature.delta(mfccs)
        
        # MFCC delta-deltas (acceleration)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Aggregate MFCCs
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta, axis=1)
        features['mfcc_delta_std'] = np.std(mfcc_delta, axis=1)
        features['mfcc_delta2_mean'] = np.mean(mfcc_delta2, axis=1)
        features['mfcc_delta2_std'] = np.std(mfcc_delta2, axis=1)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
        )
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
        )
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.HOP_LENGTH)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.HOP_LENGTH)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
        )
        features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
        features['spectral_contrast_std'] = np.std(spectral_contrast, axis=1)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
        )
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        return features
    
    def features_to_vector(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Flatten feature dictionary to a single feature vector.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            1D numpy array of all features concatenated
        """
        feature_list = []
        
        # Order matters - keep consistent
        ordered_keys = [
            'mfcc_mean', 'mfcc_std', 'mfcc_delta_mean', 'mfcc_delta_std',
            'mfcc_delta2_mean', 'mfcc_delta2_std',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std',
            'zcr_mean', 'zcr_std',
            'rms_mean', 'rms_std',
            'spectral_contrast_mean', 'spectral_contrast_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std'
        ]
        
        for key in ordered_keys:
            value = features[key]
            if isinstance(value, np.ndarray):
                feature_list.extend(value.flatten())
            else:
                feature_list.append(value)
        
        return np.array(feature_list)
    
    def process_audio(self, base64_audio: str, input_format: str = "mp3") -> np.ndarray:
        """
        Full pipeline: decode, convert, extract features.
        
        Args:
            base64_audio: Base64 encoded audio
            input_format: Audio format (default: mp3)
            
        Returns:
            Feature vector for classification
        """
        # Decode base64
        audio_bytes = self.decode_base64_audio(base64_audio)
        
        # Convert to WAV and load
        audio, sr = self.convert_to_wav(audio_bytes, input_format)
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        # Convert to vector
        feature_vector = self.features_to_vector(features)
        
        return feature_vector


# Singleton instance
audio_processor = AudioProcessor()
