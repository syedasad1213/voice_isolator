# Voice Isolator

## Introduction

**Voice Isolator** is a professional speech enhancement tool designed to isolate and clean voice audio files using advanced denoising models. It provides a command-line interface for processing single files, batch directories, and running synthetic tests.

---

## File Overview

- ```**src/cli/main.py**```  
  Main CLI entry point. Handles user commands for processing, batch jobs, and testing.

- ```**src/core/audio_processor.py**```  
  Contains the logic for processing audio files using the selected model.

- ```**src/models/facebook_deioniser.py**```  
    Implements the Facebook denoiser model used for voice isolation.

- ```**data/**```   
    Directory for the input audio files, there are already two sample audio files before processing.

- ```**output/**```  
  Directory where cleaned audio files are saved, you can already see the cleaned versions of input sample audios in this.

---

## How to Clone

Clone the repository using Git:

```
git clone https://github.com/syedasad1213/voice_isolator.git
cd voice_isolator
```

---

## Installation

Install required Python packages:

```
pip install click numpy soundfile
```

---

## How to Navigate

- All source code is in the `src/` directory.
- Place your audio files in the `data/` directory.
- Processed files will be saved in the `output/` directory.

---

## How to Work With Voice Isolator

### 1. Single File Processing

Process one audio file:

```
python -m src.cli.main process <input_file> <output_file>
```

Example:
```
python -m src.cli.main process data/sample.wav output/cleaned.wav
```

### 2. Batch Processing

Process all supported audio files in a directory:

```
python -m src.cli.main batch <input_dir> <output_dir>
```

Example:
```
python -m src.cli.main batch data/ output/
```

### 3. Test Mode

Run a synthetic test to verify the system:

```
python -m src.cli.main test
```

---

## Supported Audio Formats

- WAV
- MP3
- FLAC
- M4A
- OGG

---

## Troubleshooting

- Ensure you run commands from the project root directory.
- Use valid paths for input and output.
- If you get import errors, use the `-m` flag as shown above.

---

## Contributing

Feel free to fork the repository and submit pull requests. For major changes, open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License.

---

## Author

Developed by Asad Syed

---
