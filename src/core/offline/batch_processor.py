"""Offline batch processing for high-quality enhancement."""

import os
import logging
from typing import List, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from core.audio_processor import AudioProcessor
from utils.file_utils import FileUtils
from utils.metrics import AudioMetrics

class BatchProcessor:
    def __init__(self, audio_processor: AudioProcessor, num_workers: int = 4):
        self.audio_processor = audio_processor
        self.num_workers = num_workers
        self.file_utils = FileUtils()
        self.metrics = AudioMetrics()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_single_file(self, input_path: Path, output_path: Path) -> dict:
        """Process a single audio file."""
        try:
            # Load audio
            audio, sr = self.audio_processor.load_audio(input_path)
            original_duration = len(audio) / sr
            
            # Preprocess
            preprocessed = self.audio_processor.preprocess_audio(audio)
            
            # Enhance
            enhanced = self.audio_processor.enhance_audio(preprocessed)
            
            # Save result
            self.audio_processor.save_audio(enhanced, output_path, sr)
            
            # Calculate metrics
            metrics = self.metrics.calculate_enhancement_metrics(
                original=audio, 
                enhanced=enhanced, 
                sample_rate=sr
            )
            
            return {
                'input_file': str(input_path),
                'output_file': str(output_path),
                'duration': original_duration,
                'status': 'success',
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process {input_path}: {e}")
            return {
                'input_file': str(input_path),
                'output_file': str(output_path),
                'status': 'failed',
                'error': str(e)
            }
    
    def process_directory(self, input_dir: Union[str, Path], 
                         output_dir: Union[str, Path],
                         recursive: bool = True) -> List[dict]:
        """Process all audio files in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = self.file_utils.find_audio_files(input_dir, recursive)
        
        if not audio_files:
            self.logger.warning(f"No audio files found in {input_dir}")
            return []
        
        self.logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Prepare processing tasks
        tasks = []
        for input_file in audio_files:
            # Maintain directory structure in output
            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_suffix('.wav')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            tasks.append((input_file, output_file))
        
        # Process files in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.process_single_file, input_path, output_path): 
                (input_path, output_path) for input_path, output_path in tasks
            }
            
            # Process completed tasks
            with tqdm(total=len(tasks), desc="Processing files") as pbar:
                for future in as_completed(future_to_task):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    # Log progress
                    if result['status'] == 'success':
                        self.logger.info(f"✓ {result['input_file']}")
                    else:
                        self.logger.error(f"✗ {result['input_file']}: {result.get('error', 'Unknown error')}")
        
        return results
    
    def generate_report(self, results: List[dict], output_file: str = None) -> str:
        """Generate processing report."""
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        report = f"""
# Voice Isolator Batch Processing Report

## Summary
- Total files processed: {len(results)}
- Successful: {len(successful)}
- Failed: {len(failed)}
- Success rate: {len(successful)/len(results)*100:.1f}%

## Processing Statistics
"""
        
        if successful:
            total_duration = sum(r['duration'] for r in successful)
            avg_snr_improvement = np.mean([r['metrics']['snr_improvement'] 
                                         for r in successful if 'metrics' in r])
            
            report += f"""
- Total audio duration processed: {total_duration:.1f} seconds
- Average SNR improvement: {avg_snr_improvement:.2f} dB
"""
        
        if failed:
            report += f"""
## Failed Files
"""
            for result in failed:
                report += f"- {result['input_file']}: {result.get('error', 'Unknown error')}\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report