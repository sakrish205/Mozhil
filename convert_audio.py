"""
Audio to Base64 Converter
Drag and drop your audio file or enter the path to convert it.
"""

import base64
import os
import sys

def convert_audio_to_base64(audio_path: str, output_path: str = "Audio Base64 Format.txt"):
    """Convert an audio file to Base64 and save to a text file."""
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"❌ Error: File not found: {audio_path}")
        return False
    
    # Get file extension
    ext = os.path.splitext(audio_path)[1].lower()
    print(f"📁 File: {audio_path}")
    print(f"📝 Format: {ext}")
    
    try:
        # Read the audio file
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        # Convert to Base64
        base64_data = base64.b64encode(audio_bytes).decode()
        
        # Save to output file
        with open(output_path, "w") as f:
            f.write(base64_data)
        
        print(f"✅ Success! Saved to: {output_path}")
        print(f"📊 File size: {len(audio_bytes)} bytes")
        print(f"📊 Base64 length: {len(base64_data)} characters")
        print(f"\n💡 Use 'Audio Format': '{ext[1:]}' in the competition form")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("🎤 Audio to Base64 Converter")
    print("=" * 50)
    
    # Get audio file path
    if len(sys.argv) > 1:
        # File path passed as argument (drag and drop)
        audio_path = sys.argv[1]
    else:
        # Ask user for input
        print("\nEnter the path to your audio file (MP3, WAV, etc.):")
        audio_path = input("> ").strip().strip('"')
    
    # Convert
    if audio_path:
        convert_audio_to_base64(audio_path)
    else:
        print("❌ No file path provided")
    
    print("\nPress Enter to exit...")
    input()
