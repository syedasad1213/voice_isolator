"""Base model interface."""

from abc import ABC, abstractmethod
import numpy as np

class BaseEnhancementModel(ABC):
    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        pass
    
    def get_model_info(self) -> dict:
        return {'name': 'Base Model'}