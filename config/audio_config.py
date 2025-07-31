"""Audio processing configuration settings."""

class AudioConfig:
    # Audio format settings
    SAMPLE_RATE = 16000  # Standard for speech processing
    CHANNELS = 1  # Mono for voice isolation
    BIT_DEPTH = 16
    
    # Processing parameters
    FRAME_SIZE = 512  # Samples per frame
    HOP_LENGTH = 256  # Overlap between frames
    WINDOW_TYPE = 'hann'
    
    # Supported formats
    SUPPORTED_INPUT_FORMATS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    SUPPORTED_OUTPUT_FORMATS = ['.wav', '.flac']
    DEFAULT_OUTPUT_FORMAT = '.wav'
    
    # Quality settings
    NOISE_REDUCTION_STRENGTH = 0.8
    VOICE_ENHANCEMENT_GAIN = 1.2
    SPECTRAL_GATE_THRESHOLD = 0.03