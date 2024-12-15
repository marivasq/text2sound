import os
import csv
import librosa
import soundfile as sf

# Define paths
DATA_FOLDER = "data"
RAW_AUDIO_FOLDER = os.path.join(DATA_FOLDER, "raw")
PROCESSED_AUDIO_FOLDER = os.path.join(DATA_FOLDER, "processed")
METADATA_FILE = os.path.join(DATA_FOLDER, "metadata.csv")

# Ensure processed audio folder exists
os.makedirs(PROCESSED_AUDIO_FOLDER, exist_ok=True)

def denoise_and_trim_silence(file_path):
    # Load audio file
    waveform, sr = librosa.load(file_path, sr=None) # set sr to None to use the original sample rate

    return

def normalize_audio(file_path, output_path): # TODO: fix later
    """Normalize audio to -1 to 1 range and save it."""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=None)
        
        # Normalize (loudness)
        max_amplitude = max(abs(audio))
        normalized_audio = audio / max_amplitude if max_amplitude > 0 else audio
        
        # Save normalized audio
        sf.write(output_path, normalized_audio, sr)
        print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_dataset():
    """Preprocess all audio files and update metadata."""
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
        file_name, category, tempo, tags, num_downloads, duration, license_url, username = row
        raw_audio_path = os.path.join(RAW_AUDIO_FOLDER, file_name)
        processed_audio_path = os.path.join(PROCESSED_AUDIO_FOLDER, file_name)

        # Process audio if it exists
        if os.path.exists(raw_audio_path):
            #normalize_audio(raw_audio_path, processed_audio_path)

            # Add processed file path to metadata
            updated_rows.append([file_name, category, tempo, tags, num_downloads, duration, license_url, username])
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
