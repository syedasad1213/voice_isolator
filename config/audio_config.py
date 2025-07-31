"""Audio processing configuration."""

class AudioConfig:
    SAMPLE_RATE = 16000
    CHANNELS = 1
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.m4a']
    
    NOISE_REDUCTION_STRENGTH = 0.8
    VOICE_ENHANCEMENT_GAIN = 1.2