"""Real-time processing configuration."""

class RealtimeConfig:
    # Buffer settings
    BUFFER_SIZE = 1024  # Samples
    NUM_BUFFERS = 4
    MAX_LATENCY_MS = 50  # Target latency
    
    # Processing settings
    PROCESSING_CHUNK_SIZE = 512
    OVERLAP_FACTOR = 0.5
    
    # Device settings
    DEFAULT_INPUT_DEVICE = None  # Auto-detect
    DEFAULT_OUTPUT_DEVICE = None
    
    # Performance settings
    USE_GPU = True
    NUM_THREADS = 4
    PRIORITY_MODE = True  # High priority processing