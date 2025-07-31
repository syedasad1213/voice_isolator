"""Main audio processing engine."""

import numpy as np
import librosa
import soundfile as sf
from typing import Union, Tuple, Optional
from pathlib import Path

from models.base_model import BaseEnhancementModel
from utils.audio_utils import AudioUtils
from config.audio_config import AudioConfig

class AudioProcessor:
    def __init__(self, model: BaseEnhancementModel, config: AudioConfig = None):
        self.model = model
        self.config = config or AudioConfig()
        self.audio_utils = AudioUtils()
        
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file with proper preprocessing."""
        try:
            audio, sr = librosa.load(
                file_path, 
                sr=self.config.SAMPLE_RATE,
                mono=True
            )
            return audio, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize and prepare audio for processing."""
        # Normalize amplitude
        audio = librosa.util.normalize(audio)
        
        # Apply pre-emphasis filter
        audio = self.audio_utils.pre_emphasis(audio)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        return audio
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply voice enhancement and noise reduction."""
        # Process through the selected model
        enhanced = self.model.process(audio)
        
        # Post-processing
        enhanced = self.postprocess_audio(enhanced)
        
        return enhanced
    
    def postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply final enhancement and normalization."""
        # Apply voice enhancement gain
        audio = audio * self.config.VOICE_ENHANCEMENT_GAIN
        
        # Final normalization to prevent clipping
        audio = np.clip(audio, -1.0, 1.0)
        
        # Apply de-emphasis filter
        audio = self.audio_utils.de_emphasis(audio)
        
        return audio
    
    def save_audio(self, audio: np.ndarray, output_path: Union[str, Path], 
                   sample_rate: int = None) -> None:
        """Save processed audio to file."""
        sr = sample_rate or self.config.SAMPLE_RATE
        
        try:
            sf.write(output_path, audio, sr)
        except Exception as e:
            raise ValueError(f"Failed to save audio file: {e}")