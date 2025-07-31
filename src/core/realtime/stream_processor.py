"""Real-time audio stream processing."""

import numpy as np
import pyaudio
import threading
from collections import deque
from typing import Callable, Optional

from core.audio_processor import AudioProcessor
from utils.signal_processing import SignalProcessor
from config.realtime_config import RealtimeConfig

class StreamProcessor:
    def __init__(self, audio_processor: AudioProcessor, 
                 config: RealtimeConfig = None):
        self.audio_processor = audio_processor
        self.config = config or RealtimeConfig()
        self.signal_processor = SignalProcessor()
        
        # Audio stream setup
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.is_processing = False
        
        # Buffer management
        self.input_buffer = deque(maxlen=self.config.NUM_BUFFERS)
        self.output_buffer = deque(maxlen=self.config.NUM_BUFFERS)
        
        # Threading
        self.processing_thread = None
        self.buffer_lock = threading.Lock()
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for real-time processing."""
        if status:
            print(f"Audio callback status: {status}")
            
        # Convert input data to numpy array
        input_audio = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to input buffer
        with self.buffer_lock:
            self.input_buffer.append(input_audio)
        
        # Get processed audio from output buffer
        try:
            with self.buffer_lock:
                output_audio = self.output_buffer.popleft()
        except IndexError:
            # If no processed audio available, return zeros
            output_audio = np.zeros(frame_count, dtype=np.float32)
        
        return (output_audio.tobytes(), pyaudio.paContinue)
    
    def processing_worker(self):
        """Background thread for audio processing."""
        while self.is_processing:
            try:
                # Get audio from input buffer
                with self.buffer_lock:
                    if not self.input_buffer:
                        continue
                    input_audio = self.input_buffer.popleft()
                
                # Process audio
                enhanced_audio = self.audio_processor.enhance_audio(input_audio)
                
                # Add to output buffer
                with self.buffer_lock:
                    self.output_buffer.append(enhanced_audio)
                    
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def start_stream(self, input_device: Optional[int] = None,
                    output_device: Optional[int] = None) -> None:
        """Start real-time audio processing stream."""
        self.stream = self.pyaudio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.audio_processor.config.SAMPLE_RATE,
            input=True,
            output=True,
            input_device_index=input_device,
            output_device_index=output_device,
            frames_per_buffer=self.config.BUFFER_SIZE,
            stream_callback=self.audio_callback
        )
        
        # Start processing thread
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.start()
        
        self.stream.start_stream()
        print("Real-time voice isolation started...")
    
    def stop_stream(self) -> None:
        """Stop real-time processing."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
        
        print("Real-time processing stopped.")
    
    def __del__(self):
        self.stop_stream()
        self.pyaudio.terminate()