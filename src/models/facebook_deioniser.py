"""Facebook Denoiser model wrapper."""

import torch
import numpy as np
from .base_model import BaseEnhancementModel

class FacebookDenoiser(BaseEnhancementModel):
    def __init__(self, device='auto'):
        super().__init__()
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading Facebook Denoiser on {self.device}...")
        
        try:
            # Load the pretrained DNS64 model
            self.model = torch.hub.load(
                'facebookresearch/denoiser:main', 
                'dns64',
                pretrained=True
            )
            self.model.to(self.device)
            self.model.eval()
            print("Facebook Denoiser loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load Facebook Denoiser: {e}")
            print("Falling back to simple spectral subtraction...")
            self.model = None
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through Facebook Denoiser."""
        
        if self.model is None:
            # Fallback to simple processing
            return self._simple_denoise(audio)
        
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            
            # Process
            with torch.no_grad():
                enhanced = self.model(audio_tensor)
            
            # Convert back to numpy
            enhanced_audio = enhanced.squeeze().cpu().numpy()
            return enhanced_audio
            
        except Exception as e:
            print(f"Error in processing: {e}")
            return self._simple_denoise(audio)
    
    def _simple_denoise(self, audio):
        """Simple fallback denoising."""
        import librosa
        
        # Basic spectral subtraction
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Noise estimation from first 10% of frames
        noise_frames = max(1, magnitude.shape[1] // 10)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Spectral subtraction
        alpha = 2.0
        beta = 0.01
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft)
        
        return enhanced_audio
    
    def get_model_info(self):
        return {
            'name': 'Facebook Denoiser (DNS64)',
            'device': self.device,
            'loaded': self.model is not None
        }