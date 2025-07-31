"""Command line interface for Voice Isolator."""

import argparse
import sys
from pathlib import Path

from core.audio_processor import AudioProcessor
from models.facebook_denoiser import FacebookDenoiser
from offline.batch_processor import BatchProcessor
from realtime.stream_processor import StreamProcessor

class VoiceIsolatorCLI:
    def __init__(self):
        self.parser = self.create_parser()
    
    def create_parser(self):
        parser = argparse.ArgumentParser(
            description="Voice Isolator - Real-time Speech Enhancement System"
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Offline processing
        offline_parser = subparsers.add_parser('offline', help='Process audio files')
        offline_parser.add_argument('input', help='Input file or directory')
        offline_parser.add_argument('output', help='Output file or directory')
        offline_parser.add_argument('--model', default='facebook_denoiser', 
                                  choices=['facebook_denoiser', 'demucs', 'rnnoise'],
                                  help='Enhancement model to use')
        offline_parser.add_argument('--recursive', '-r', action='store_true',
                                  help='Process directories recursively')
        offline_parser.add_argument('--workers', '-w', type=int, default=4,
                                  help='Number of parallel workers')
        
        # Real-time processing
        realtime_parser = subparsers.add_parser('realtime', help='Real-time processing')
        realtime_parser.add_argument('--model', default='facebook_denoiser',
                                   choices=['facebook_denoiser', 'rnnoise'],
                                   help='Real-time model to use')
        realtime_parser.add_argument('--input-device', type=int,
                                   help='Input audio device index')
        realtime_parser.add_argument('--output-device', type=int,
                                   help='Output audio device index')
        realtime_parser.add_argument('--list-devices', action='store_true',
                                   help='List available audio devices')
        
        return parser
    
    def run(self):
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        if args.command == 'offline':
            self.run_offline(args)
        elif args.command == 'realtime':
            self.run_realtime(args)
    
    def run_offline(self, args):
        print(f"Starting offline processing with {args.model}...")
        
        # Initialize model
        if args.model == 'facebook_denoiser':
            model = FacebookDenoiser()
        # Add other model initializations
        
        # Initialize processor
        audio_processor = AudioProcessor(model)
        batch_processor = BatchProcessor(audio_processor, args.workers)
        
        # Process files
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if input_path.is_file():
            result = batch_processor.process_single_file(input_path, output_path)
            print(f"Processing complete: {result['status']}")
        else:
            results = batch_processor.process_directory(
                input_path, output_path, args.recursive
            )
            report = batch_processor.generate_report(results)
            print(report)
    
    def run_realtime(self, args):
        if args.list_devices:
            self.list_audio_devices()
            return
        
        print(f"Starting real-time processing with {args.model}...")
        
        # Initialize model and processor
        if args.model == 'facebook_denoiser':
            model = FacebookDenoiser()
        
        audio_processor = AudioProcessor(model)
        stream_processor = StreamProcessor(audio_processor)
        
        try:
            stream_processor.start_stream(args.input_device, args.output_device)
            input("Press Enter to stop...")
        except KeyboardInterrupt:
            pass
        finally:
            stream_processor.stop_stream()
    
    def list_audio_devices(self):
        import pyaudio
        p = pyaudio.PyAudio()
        
        print("Available audio devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(f"{i}: {info['name']} (In: {info['maxInputChannels']}, Out: {info['maxOutputChannels']})")
        
        p.terminate()

if __name__ == '__main__':
    cli = VoiceIsolatorCLI()
    cli.run()