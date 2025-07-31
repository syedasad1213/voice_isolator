"""Facebook Denoiser model wrapper."""

import torch
import torchaudio
import numpy as np
from pathlib import Path

from .base_model import BaseEnhancementModel

class FacebookDenoiser(BaseEnhancementModel):
    def __init__(self, model_path: str = None, device: str = 'auto'):
        super().__init__()
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load pre-trained model
        if model_path is None:
            # Load from torch hub
            self.model = torch.hub.load(
                'facebookresearch/denoiser', 
                'dns64', 
                pretrained=True
            )
        else:
            self.model = torch.load(model_path, map_location=self.device)
            
        self.model.to(self.device)
        self.model.eval()
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through Facebook Denoiser."""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        # Process through model
        with torch.no_grad():
            enhanced = self.model(audio_tensor)
        
        # Convert back to numpy
        enhanced_audio = enhanced.squeeze().cpu().numpy()
        
        return enhanced_audio
    
    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            'name': 'Facebook Denoiser',
            'type': 'CNN-based',
            'real_time': True,
            'device': self.device,
            'parameters': sum(p.numel() for p in self.model.parameters())
        }