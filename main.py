import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import soundfile as sf
from IPython.display import Audio, display
import noisereduce as nr  # Make sure to install this library: `pip install noisereduce`

# Bandpass filter function
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Bandpass filtering function
def bandpass_filter(audio, lowcut, highcut, sr, order=5):
    b, a = butter_bandpass(lowcut, highcut, sr, order)
    return lfilter(b, a, audio)

# Function to preprocess audio files
def preprocess_audio(file_path, output_audio_path):
    try:
        # Load the audio file
        audio, sr = librosa.load(file_path, sr=22050)
        print("Audio loaded successfully.")

        # Plot the original audio waveform with a unified color palette
        plt.figure(figsize=(12, 6))
        librosa.display.waveshow(audio, sr=sr, alpha=0.8, color='teal', linewidth=2)
        plt.title("Original Audio Waveform", fontsize=16, color='midnightblue')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.grid(True, which='both', linestyle='--', color='lightgrey', alpha=0.5)
        plt.show()

        # Display the original audio player
        display(Audio(audio, rate=sr))

        # Convert to mono
        audio = librosa.to_mono(audio)
        print("Converted to mono.")

        # Apply noise reduction
        audio_denoised = nr.reduce_noise(y=audio, sr=sr)
        print("Noise reduction applied.")

        # Apply bandpass filter (e.g., between 300 Hz and 3000 Hz)
        lowcut = 300.0
        highcut = 3000.0
        audio_filtered = bandpass_filter(audio_denoised, lowcut, highcut, sr)
        print(f"Bandpass filter applied: {lowcut} Hz - {highcut} Hz.")

        # Plot the denoised and filtered audio waveform with a consistent color scheme
        plt.figure(figsize=(12, 6))
        librosa.display.waveshow(audio_filtered, sr=sr, alpha=0.8, color='royalblue', linewidth=2)
        plt.title("Processed Audio Waveform (Denoised & Bandpass Filtered)", fontsize=16, color='navy')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.grid(True, which='both', linestyle='--', color='lightgrey', alpha=0.5)
        plt.show()

        # Pitch Extraction
        pitches, magnitudes = librosa.core.piptrack(y=audio_filtered, sr=sr)
        pitch = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch.append(pitches[index, t])
        pitch = np.array(pitch)
        print(f"Pitch extracted: {pitch[:100]}...")  # Display first 100 values of pitch for brevity

        # Plot Pitch Extraction with consistent colors
        plt.figure(figsize=(12, 6))
        plt.plot(pitch, label="Extracted Pitch", color='cadetblue', linewidth=2)
        plt.title("Pitch Extraction", fontsize=16, color='darkslateblue')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Pitch (Hz)", fontsize=12)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, which='both', linestyle='--', color='lightgrey', alpha=0.5)
        plt.show()

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio_filtered)[0]
        print(f"Zero-Crossing Rate extracted: {zcr[:100]}...")  # Display first 100 values for brevity

 #helping to distinguish between voiced and unvoiced sounds, segment audio,
 #classify speech vs. music, and detect noise.
        # Plot Zero-Crossing Rate with consistent color scheme
        plt.figure(figsize=(12, 6))
        plt.plot(zcr, label="Zero-Crossing Rate", color='steelblue', linewidth=2)
        plt.title("Zero-Crossing Rate", fontsize=16, color='midnightblue')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Zero-Crossing Rate", fontsize=12)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, which='both', linestyle='--', color='lightgrey', alpha=0.5)
        plt.show()

        # Silence removal (Voice Activity Detection - VAD)
        audio_vad, _ = librosa.effects.trim(audio_filtered)
        print(f"Silence removed (VAD applied).")

        # Plot original and VAD audio waveforms with consistent colors
        plt.figure(figsize=(12, 6))
        librosa.display.waveshow(audio_filtered, sr=sr, alpha=0.8, color='darkcyan', linewidth=2)
        plt.title("Audio Before Silence Removal (VAD)", fontsize=16, color='teal')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)

        plt.figure(figsize=(12, 6))
        librosa.display.waveshow(audio_vad, sr=sr, alpha=0.8, color='mediumslateblue', linewidth=2)
        plt.title("Audio After Silence Removal (VAD)", fontsize=16, color='slateblue')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.show()

        # Normalize the audio
        audio_normalized = librosa.util.normalize(audio_vad)
        print("Audio normalized.")

        # Display the normalized audio with consistent colors
        display(Audio(audio_normalized, rate=sr))

        # Plot the original and normalized audio waveforms for comparison
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        librosa.display.waveshow(audio_vad, sr=sr, alpha=0.8, color='slategray', linewidth=2)
        plt.title("Audio Before Normalization", fontsize=16, color='darkgray')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)

        plt.subplot(1, 2, 2)
        librosa.display.waveshow(audio_normalized, sr=sr, alpha=0.8, color='mediumseagreen', linewidth=2)
        plt.title("Audio After Normalization", fontsize=16, color='darkgreen')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)

        plt.tight_layout()
        plt.figtext(0.15, 0.05, f'Original Audio Range: [{np.min(audio_vad):.4f}, {np.max(audio_vad):.4f}]', ha='left', fontsize=10, color='darkred')
        plt.figtext(0.65, 0.05, f'Normalized Audio Range: [{np.min(audio_normalized):.4f}, {np.max(audio_normalized):.4f}]', ha='left', fontsize=10, color='darkblue')
        plt.show()

        # Plot the amplitude scaling before and after normalization with gradient color
        plt.figure(figsize=(12, 6))
        plt.plot(audio_vad[:5000], label="Before Normalization", color='darkblue', alpha=0.7, linewidth=2)
        plt.plot(audio_normalized[:5000], label="After Normalization", color='red', alpha=0.7, linewidth=2)
        plt.title("Amplitude Scaling: Before and After Normalization", fontsize=16, color='darkred')
        plt.xlabel("Samples", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        # Extract MFCCs
        mfccs = np.mean(librosa.feature.mfcc(y=audio_normalized, sr=sr, n_mfcc=13), axis=1).tolist()
        print("MFCCs extracted.")

        # Plot Mel-spectrogram with a cohesive color map
        # Mel-spectrogram represents how the power (intensity)
        # of different frequencies in the signal changes over time
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_normalized, sr=sr, n_mels=128, fmax=8000)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        plt.figure(figsize=(12, 6))
        librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel', sr=sr, cmap='coolwarm')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Processed Mel-spectrogram', fontsize=16, color='darkblue')
        plt.show()

        # Save the processed audio
        sf.write(output_audio_path, audio_normalized, sr)
        print(f"Processed audio saved to {output_audio_path}")

        return {"mfccs": mfccs, "sample_rate": sr, "audio": audio_normalized, "pitch": pitch, "zcr": zcr}

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# Main script
if __name__ == "__main__":
    # Input and output audio file paths
    audio_file = "/content/Post_Malone_-_Sunflower_ft_Swae_Lee_HQ.wav"  # Replace with your file path
    output_audio_path = "processed_audio.wav"

    # Preprocess the audio file
    preprocessed = preprocess_audio(audio_file, output_audio_path)
    if preprocessed:
        mfccs = preprocessed["mfccs"]
        sr = preprocessed["sample_rate"]
        audio = preprocessed["audio"]
        pitch = preprocessed["pitch"]
        zcr = preprocessed["zcr"]

        # Print extracted MFCCs, pitch, and ZCR
        print(f"MFCCs: {mfccs}")
        print(f"Pitch: {pitch[:100]}...")  # Display first 100 values of pitch for brevity
        print(f"Zero-Crossing Rate: {zcr[:100]}...")  # Display first 100 values of ZCR for brevity

        # Play the processed audio
        print("Playing processed audio...")
        display(Audio(audio, rate=sr))
    else:
        print("Failed to preprocess the audio file.")
