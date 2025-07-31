"""Command line interface - THIS IS WHAT YOU RUN."""

import click
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ..models.facebook_deioniser import FacebookDenoiser
from ..core.audio_processor import AudioProcessor

@click.group()
def cli():
    """Voice Isolator - Professional Speech Enhancement System"""
    pass

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--model', default='facebook_denoiser', help='Model to use')
def process(input_path, output_path, model):
    """Process a single audio file."""
    
    print("Voice Isolator Pro - Starting...")
    
    # Initialize model
    if model == 'facebook_denoiser':
        model_instance = FacebookDenoiser()
    else:
        click.echo(f"Unknown model: {model}")
        return
    
    # Initialize processor
    processor = AudioProcessor(model_instance)
    
    # Process file
    try:
        result = processor.process_file(input_path, output_path)
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETE!")
        print(f"Input: {result['input_file']}")
        print(f"Output: {result['output_file']}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Processing Time: {result['processing_time']:.2f}s") 
        print(f"Real-time Factor: {result['real_time_factor']:.2f}x")
        print("="*50)
        
    except Exception as e:
        click.echo(f"Error: {e}")

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def batch(input_dir, output_dir):
    """Process all audio files in a directory."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        click.echo("No audio files found!")
        return
    
    click.echo(f"Found {len(audio_files)} audio files")
    
    # Initialize model and processor
    model_instance = FacebookDenoiser()
    processor = AudioProcessor(model_instance)
    
    # Process each file
    results = []
    for audio_file in audio_files:
        output_file = output_path / f"{audio_file.stem}_cleaned.wav"
        
        try:
            result = processor.process_file(str(audio_file), str(output_file))
            results.append(result)
        except Exception as e:
            click.echo(f"Failed to process {audio_file}: {e}")
    
    # Summary
    successful = len([r for r in results if r['status'] == 'success'])
    click.echo(f"\n Batch processing complete!")
    click.echo(f"Successfully processed: {successful}/{len(audio_files)} files")

@cli.command()
def test():
    """Test the system with generated audio."""
    
    print("Testing Voice Isolator...")
    
    # Create test audio
    import numpy as np
    import soundfile as sf
    
    # Generate test signal
    duration = 5
    sample_rate = 16000
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Voice-like signal
    voice = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
    
    # Add noise
    noise = np.random.normal(0, 0.3, len(voice))
    noisy_audio = 0.7 * voice + 0.3 * noise
    
    # Save test file
    test_input = "test_noisy.wav"
    test_output = "test_cleaned.wav"
    
    sf.write(test_input, noisy_audio, sample_rate)
    print(f"Created test file: {test_input}")
    
    # Process it
    model_instance = FacebookDenoiser()
    processor = AudioProcessor(model_instance)
    
    result = processor.process_file(test_input, test_output)
    
    print(f"\n Test complete!")
    print(f"Listen to {test_input} (noisy) vs {test_output} (cleaned)")

if __name__ == '__main__':
    cli()