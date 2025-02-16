"""
Preprocess the raw data and metadata.
"""

import os
import csv
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from sklearn.feature_extraction.text import TfidfVectorizer

# Get the directory of the currently running script
current_script_dir = os.path.dirname(__file__)

# Define paths
BASE_DIR = os.path.join(current_script_dir, 'dataset')
SPECTOGRAM_DIR = os.path.join(BASE_DIR, 'spectograms')
EMBEDDING_DIR = os.path.join(BASE_DIR, 'embeddings')

RAW_AUDIO_FOLDER = os.path.join(BASE_DIR, 'raw')
RAW_METADATA_FILE = os.path.join(RAW_AUDIO_FOLDER, 'raw_metadata.csv')

PROCESSED_AUDIO_FOLDER = os.path.join(BASE_DIR, 'processed')
PROCESSED_METADATA_FILE = os.path.join(PROCESSED_AUDIO_FOLDER, 'processed_metadata.csv')


# Ensure processed audio folder exists
os.makedirs(PROCESSED_AUDIO_FOLDER, exist_ok=True)

def denoise_and_trim_silence(y, sr, noise_duration=0.5, top_db=20):
    """
    Preprocess audio by dynamically denoising and trimming silence.

    Args:
        y (numpy.ndarray): waveform
        sr (int): sample rate or samples per second
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

    Args:
        y (numpy.ndarray): waveform
        sr (int): sample rate
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

    Args:
        y (numpy.ndarray): waveform
        sr (int): sample rate
    """
    # Resample to target sampling rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        print(f"Resampled from {sr} Hz to {target_sr} Hz")
    else:
        print(f"Audio is already at target sampling rate of {target_sr} Hz")

    return y, target_sr

def fix_audio_length(y, sr, target_length=2.0):
    """
    Standardize the length of the audio files.
    """
    target_samples = int(sr * target_length)
    if len(y) > target_samples:
        return y[:target_samples]
    else:
        return librosa.util.fix_length(y, size=target_samples)

def audio_to_spectrogram(audio, sr, n_fft=2048, hop_length=512):
    """
    Use Short-Time Fourier Transform (STFT) to convert the audio signals into spectrograms.
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    return log_spectrogram

def save_spectrograms(output_dir, spectrograms):
    """
    Save each spectrogram as a .npy file to speed up data loading during training.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, spec in enumerate(spectrograms):
        filepath = os.path.join(output_dir, f"spec_{i}.npy")
        np.save(filepath, spec)

def save_text_embedding(output_dir, text):
    """
    Tokenize the text descriptions (and tags) and convert them into word embeddings 
    (e.g., using pre-trained models like Word2Vec, GloVe, or BERT).
    """
    os.makedirs(output_dir, exist_ok=True)

    vectorizer = TfidfVectorizer()
    
    # Fit and transform the entire list of texts at once
    text_features = vectorizer.fit_transform(text).toarray()

    # Save each row (document embedding) separately
    for i, features in enumerate(text_features):
        filepath = os.path.join(output_dir, f"embedding_{i}.npy")
        np.save(filepath, features)

def more():
    # Execute dynamic range compression, pitch/tempo standardization, and other adjustments ??
    return

def save_processed_file_by_format(path, y, sr):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        file_format = 'WAV'
    elif ext == '.aif' or ext == '.aiff':
        file_format = 'AIFF'
    elif ext == '.flac':
        file_format = 'FLAC'
    else:
        print(path)
        raise ValueError(f"Unsupported file extension: {ext}")

    sf.write(path, y, sr, format=file_format)
    return

def process_dataset():
    """
    Preprocess all audio files and update metadata.
    """
    if not os.path.exists(RAW_METADATA_FILE):
        print("Metadata file not found. Exiting.")
        return

    # Load metadata
    with open(RAW_METADATA_FILE, mode='r', encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader)

    # Prepare new metadata entries
    updated_rows = []
    spectrograms = []
    text_embeddings = []
    for row in rows:
        file_name, sound_id, category, tempo, tags, description, num_downloads, duration, license_url, username = row
        PROCESSED_SUBFOLDER = os.path.join(PROCESSED_AUDIO_FOLDER, category)
        processed_audio_path = os.path.join(PROCESSED_SUBFOLDER, file_name)
        raw_audio_path = os.path.join(RAW_AUDIO_FOLDER, category, file_name)

        # Process audio if it exists
        if os.path.exists(raw_audio_path):

            # Load audio file
            audio, sr = librosa.load(raw_audio_path, sr=None) # set sr to None to use the original sample rate

            audio, sr = denoise_and_trim_silence(audio, sr)
            audio, sr = convert_mono_and_normalize(audio, sr)
            audio, sr = resample_audio(audio, sr)
            audio = fix_audio_length(audio, sr)

            # Make new processed subdirectory
            os.makedirs(PROCESSED_SUBFOLDER, exist_ok=True)

            # Save the processed file
            save_processed_file_by_format(processed_audio_path, audio, sr)

            # Create spectogram
            spectrogram = audio_to_spectrogram(audio, sr)
            spectrograms.append(spectrogram)

            # Create text embeddings
            text_embeddings.append(description)

            # Add processed file path to metadata
            updated_rows.append([file_name, sound_id, category, tempo, tags, description, num_downloads, duration, license_url, username])
        else:
            print(f"Raw audio file not found: {raw_audio_path}")

    # Save spectograms
    save_spectrograms(SPECTOGRAM_DIR, spectrograms)

    # Save text-embeddings
    save_text_embedding(EMBEDDING_DIR, text_embeddings)

    # Save updated metadata
    with open(PROCESSED_METADATA_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(updated_rows)
    print(f"Metadata updated and saved to {PROCESSED_METADATA_FILE}")

if __name__ == "__main__":
    process_dataset()
