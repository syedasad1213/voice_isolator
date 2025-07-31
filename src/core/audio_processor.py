"""Main audio processing engine."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Union, Tuple

class AudioProcessor:
    def __init__(self, model, sample_rate=16000):
        self.model = model
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            print(f"Loaded: {len(audio)/sr:.2f}s audio at {sr}Hz")
            return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {e}")
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:  
        """Preprocess audio."""
        # Normalize
        audio = librosa.util.normalize(audio)
        
        # Pre-emphasis
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        return audio
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Main enhancement function."""
        print("Enhancing audio...")
        enhanced = self.model.process(audio)
        return self.postprocess_audio(enhanced)
    
    def postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Post-process enhanced audio."""
        # Apply gain
        audio = audio * 1.2
        
        # De-emphasis
        from scipy import signal
        audio = signal.lfilter([1], [1, -0.97], audio)
        
        # Final normalization
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def save_audio(self, audio: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save processed audio."""
        try:
            sf.write(output_path, audio, self.sample_rate)
            print(f"Saved: {output_path}")
        except Exception as e:
            raise ValueError(f"Failed to save {output_path}: {e}")
    
    def process_file(self, input_path: str, output_path: str) -> dict:
        """Process single file end-to-end."""
        import time
        
        print(f"Processing: {input_path} -> {output_path}")
        
        start_time = time.time()
        
        # Load
        audio, sr = self.load_audio(input_path)
        original_duration = len(audio) / sr
        
        # Process
        preprocessed = self.preprocess_audio(audio)
        enhanced = self.enhance_audio(preprocessed)
        
        # Save
        self.save_audio(enhanced, output_path)
        
        processing_time = time.time() - start_time
        
        result = {
            'input_file': input_path,
            'output_file': output_path,
            'duration': original_duration,
            'processing_time': processing_time,
            'real_time_factor': original_duration / processing_time,
            'status': 'success'
        }
        
        print(f"Complete! RTF: {result['real_time_factor']:.2f}x")
        return result