Audio Preprocessing Pipeline
A comprehensive Python-based audio preprocessing pipeline for speech and music analysis. This tool applies various audio processing techniques including noise reduction, filtering, feature extraction, and visualization.

Features
Audio Loading & Conversion: Load audio files and convert to mono
Noise Reduction: Remove background noise using spectral gating
Bandpass Filtering: Filter frequencies between 300-3000 Hz (ideal for speech)
Pitch Extraction: Extract fundamental frequency using piptrack algorithm
Zero-Crossing Rate Analysis: Analyze signal characteristics for voice activity detection
Voice Activity Detection (VAD): Automatic silence removal
Audio Normalization: Normalize amplitude levels
Feature Extraction: Extract MFCC features for machine learning applications
Visualization: Generate comprehensive plots and spectrograms
Audio Playback: Interactive audio players for before/after comparison
Requirements
Dependencies
bash
pip install librosa
pip install numpy
pip install matplotlib
pip install scipy
pip install soundfile
pip install noisereduce
pip install IPython
System Requirements
Python 3.7+
Jupyter Notebook (recommended for interactive features)
Audio codec support (handled by librosa and soundfile)
Installation
Clone this repository:
bash
git clone https://github.com/yourusername/audio-preprocessing-pipeline.git
cd audio-preprocessing-pipeline
Install required packages:
bash
pip install -r requirements.txt
Usage
Basic Usage
python
from audio_preprocessing import preprocess_audio

# Process an audio file
audio_file = "path/to/your/audio.wav"
output_path = "processed_audio.wav"

result = preprocess_audio(audio_file, output_path)

if result:
    print("Processing completed successfully!")
    print(f"MFCCs: {result['mfccs']}")
    print(f"Sample Rate: {result['sample_rate']}")
else:
    print("Processing failed.")
Advanced Usage
The preprocessing pipeline performs the following steps automatically:

Load Audio: Loads audio at 22.05 kHz sample rate
Noise Reduction: Applies spectral noise reduction
Bandpass Filter: Filters frequencies between 300-3000 Hz
Feature Extraction: Extracts pitch and zero-crossing rate
Voice Activity Detection: Removes silence segments
Normalization: Normalizes audio amplitude
MFCC Extraction: Computes 13 MFCC coefficients
Customization
You can modify the preprocessing parameters:

python
# Modify bandpass filter range
lowcut = 300.0   # Lower frequency bound (Hz)
highcut = 3000.0 # Upper frequency bound (Hz)

# Adjust MFCC parameters
n_mfcc = 13      # Number of MFCC coefficients
Output
The pipeline generates:

Processed Audio File: Clean, normalized audio file
Feature Dictionary: Contains MFCCs, sample rate, audio data, pitch, and ZCR
Visualizations:
Original vs processed waveforms
Pitch extraction plots
Zero-crossing rate analysis
Mel-spectrogram
Before/after normalization comparison
File Structure
audio-preprocessing-pipeline/
│
├── audio_preprocessing.py    # Main preprocessing script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── examples/               # Example audio files and outputs
└── docs/                   # Additional documentation
Technical Details
Audio Processing Pipeline
Noise Reduction: Uses spectral subtraction to reduce background noise
Bandpass Filtering: Butterworth filter (5th order) for frequency selection
Pitch Extraction: Probabilistic YIN algorithm via librosa's piptrack
Zero-Crossing Rate: Measures signal sign changes for voice classification
Voice Activity Detection: Trim-based silence removal
Normalization: Peak normalization to [-1, 1] range
Feature Extraction
MFCCs: 13 coefficients representing spectral envelope
Pitch: Fundamental frequency extraction
ZCR: Zero-crossing rate for voiced/unvoiced classification
Applications
This preprocessing pipeline is suitable for:

Speech Recognition: Prepare audio for ASR systems
Music Analysis: Extract features for music information retrieval
Audio Classification: Preprocess data for ML models
Voice Activity Detection: Identify speech segments
Audio Quality Enhancement: Clean noisy recordings
Examples
Processing Speech Audio
python
# Ideal for speech processing with 300-3000 Hz bandpass
speech_result = preprocess_audio("speech_sample.wav", "clean_speech.wav")
Processing Music Audio
python
# For music, you might want to adjust the frequency range
# Modify the bandpass filter parameters in the code
music_result = preprocess_audio("music_sample.wav", "processed_music.wav")
