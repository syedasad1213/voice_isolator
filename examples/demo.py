"""Simple demo script."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.facebook_deioniser import FacebookDenoiser
from src.core.audio_processor import AudioProcessor
import numpy as np
import soundfile as sf

def main():
    print("Voice Isolator Demo")
    print("="*40)
    
    # Create test audio
    print("Creating test audio...")
    sample_rate = 16000
    duration = 3
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Simple voice + noise
    voice = np.sin(2 * np.pi * 300 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
    noise = np.random.normal(0, 0.4, len(voice))
    noisy_audio = voice + noise
    
    # Save test file
    test_input = "demo_input.wav"
    sf.write(test_input, noisy_audio, sample_rate)
    
    # Initialize system
    print("Loading Facebook Denoiser...")
    model = FacebookDenoiser()
    processor = AudioProcessor(model)
    
    # Process
    print("Processing audio...")
    result = processor.process_file(test_input, "demo_output.wav")
    
    print("\nDemo complete!")
    print("Files created:")
    print("- demo_input.wav (noisy)")
    print("- demo_output.wav (cleaned)")
    print(f"Real-time factor: {result['real_time_factor']:.2f}x")

if __name__ == "__main__":
    main()