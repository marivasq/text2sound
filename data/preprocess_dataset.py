import os
import csv
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr

# Define paths
DATA_FOLDER = "dataset"
RAW_AUDIO_FOLDER = os.path.join(DATA_FOLDER, "raw")
PROCESSED_AUDIO_FOLDER = os.path.join(DATA_FOLDER, "processed")
METADATA_FILE = os.path.join(DATA_FOLDER, "metadata.csv")

# Ensure processed audio folder exists
os.makedirs(PROCESSED_AUDIO_FOLDER, exist_ok=True)

def denoise_and_trim_silence(y, sr, noise_duration=0.5, top_db=20):
    """
    Preprocess audio by dynamically denoising and trimming silence.

    Args:
        filename (str): Path to the audio file.
        noise_duration (float): Duration in seconds for estimating noise.
        top_db (int): Threshold in decibels for trimming silence.

    Returns:
        np.ndarray: Preprocessed audio array.
        int: Sampling rate.
    """
    
    # Dynamically find a quiet region for noise profiling
    frame_length = int(sr * noise_duration)
    hop_length = frame_length // 2
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find the index of the minimum RMS energy (quietest region)
    quiet_frame_index = np.argmin(rms)
    start = quiet_frame_index * hop_length
    end = start + frame_length
    noise_sample = y[start:end]

    if len(noise_sample) < frame_length:
        print("Warning: Not enough data for noise sample, skipping denoising")
        noise_sample = y[:frame_length]  # Use the first frame as a fallback

    # Denoise using dynamically identified noise profile
    reduced_noise = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr)

    # Trim silence
    y_trimmed, _ = librosa.effects.trim(reduced_noise, top_db=top_db)

    return y_trimmed, sr

def convert_mono_and_normalize(y, sr): # TODO: fix later
    """
    Normalize audio to -1 to 1 range.
    """
    try:
        # Convert to mono if not already
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Normalize (loudness)
        normalized_y = librosa.util.normalize(y)
        
        # Save normalized audio
        return normalized_y, sr
    except Exception as e:
        print(f"Error in mono and normalize: {e}")

def resample_audio(y, sr, target_sr=16000):
    """
    Resample audio to target sample rate.
    """
    # Resample to target sampling rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        print(f"Resampled from {sr} Hz to {target_sr} Hz")
    else:
        print(f"Audio is already at target sampling rate of {target_sr} Hz")

    return y, target_sr

def more():
    # Execute dynamic range compression, pitch/tempo standardization, and other adjustments ??
    return

def process_dataset():
    """
    Preprocess all audio files and update metadata.
    """
    if not os.path.exists(METADATA_FILE):
        print("Metadata file not found. Exiting.")
        return

    # Load metadata
    with open(METADATA_FILE, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)

    # Prepare new metadata entries
    updated_rows = []
    for row in rows:
        file_name, category, tempo, tags, description, num_downloads, duration, license_url, username = row
        FILE_PATH = os.path.join(category, file_name)
        raw_audio_path = os.path.join(RAW_AUDIO_FOLDER, FILE_PATH)
        processed_audio_path = os.path.join(PROCESSED_AUDIO_FOLDER, FILE_PATH)

        # Process audio if it exists
        if os.path.exists(raw_audio_path):

            # Load audio file
            audio, sr = librosa.load(raw_audio_path, sr=None) # set sr to None to use the original sample rate

            audio, sr = denoise_and_trim_silence(audio, sr)
            audio, sr = convert_mono_and_normalize(audio, sr)
            audio, sr = resample_audio(audio, sr)

            # Save the processed file
            sf.write(processed_audio_path, audio, sr)

            # Add processed file path to metadata
            updated_rows.append([file_name, category, tempo, tags, description, num_downloads, duration, license_url, username])
        else:
            print(f"Raw audio file not found: {raw_audio_path}")

    # Save updated metadata
    updated_metadata_file = os.path.join(DATA_FOLDER, "processed_metadata.csv")
    with open(updated_metadata_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(updated_rows)
    print(f"Metadata updated and saved to {updated_metadata_file}")

if __name__ == "__main__":
    process_dataset()
