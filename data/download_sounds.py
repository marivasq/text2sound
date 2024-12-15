import os
import requests
import csv

# Set up the folder where the sounds will be saved
BASE_SAVE_DIR = os.path.join(os.getcwd(), 'data', 'dataset')

# Ensure the directory exists
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# FreeSound API settings
API_KEY = '' # Replace with your key
BASE_URL = 'https://freesound.org/apiv2/'

# Filter by license (e.g., 'cc0', 'by', 'by-sa', etc.)
ALLOWED_LICENSES = ['cc0', 'by']

# Sound categories based on tags
CATEGORY_MAPPING = {
    'kick': 'kicks',
    'snare': 'snares',
    'hihat': 'hi-hats',
    'clap': 'claps',
    'tom': 'toms',
    'cymbal': 'cymbals',
    'shaker': 'shakers',
    'bass': 'bass' # add an other category??
}

# Initialize the metadata CSV file
metadata_file = os.path.join(BASE_SAVE_DIR, 'metadata.csv')
with open(metadata_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write headers for the metadata CSV file
    writer.writerow(['filename', 'category', 'tempo', 'tags', 'description', 'num_downloads', 'duration', 'license', 'username'])


def fetch_metadata(sound_id):
    url = f'{BASE_URL}sounds/{sound_id}/'
    analysis_url = f"{BASE_URL}sounds/{sound_id}/analysis/"
    headers = {
        "Authorization": f"Token {API_KEY}"
    }

    try:
        # Fetch tempo from analysis endpoint
        analysis_response = requests.get(analysis_url, headers=headers)
        if analysis_response.status_code == 200:
            tempo = analysis_response.json().get("rhythm", {}).get("bpm", "N/A")
        else:
            tempo = "N/A"
        
        # Fetch other metadata from detail endpoint
        detail_response = requests.get(url, headers=headers)
        if detail_response.status_code == 200:
            detail_data = detail_response.json()
            # print(detail_data)
            description = detail_data.get("description", "N/A")
            num_downloads = detail_data.get("num_downloads", "N/A")
            duration = detail_data.get("duration", "N/A")
            license_url = detail_data.get("license", "N/A")
            username = detail_data.get("username", "N/A")
        else:
            description = num_downloads = duration = license_url = username = "N/A"
        
        return tempo, description, num_downloads, duration, license_url, username
    except Exception as e:
        print(f"Error fetching metadata: {e}")
    # Return default values in case of an error
    return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"


def search_sounds(query, num_results=10, license_filter=None):
    params = {
        'query': query,
        'fields': 'id,name,tags,previews,license,num_downloads',
        'token': API_KEY,
        'sort': 'num_downloads_desc',  # Sort by downloads in descending order
        'page_size': num_results  # Fetch top 'num_results' per request
    }
    response = requests.get(BASE_URL + 'search/text/', params=params)
    
    if response.status_code == 200:
        results = response.json()
        
        # Filter by license if applicable
        if license_filter:
            results['results'] = [sound for sound in results['results'] if sound['license'] in license_filter]
        
        return results['results']  # Return all results, already sorted by downloads
    else:
        print(f"Failed to fetch sounds: {response.status_code}")
    
    return []


def download_sound(sound_id, sound_name, tags, preview_url, metadata):
    # Determine the category based on tags
    category = 'others'  # Default category if no match is found
    for tag in tags:
        for key, value in CATEGORY_MAPPING.items():
            if key in tag.lower():
                category = value
                break

    # Ensure the category directory exists
    category_dir = os.path.join(BASE_SAVE_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    # Download the sound using the provided preview URL
    try:
        audio_data = requests.get(preview_url)
        audio_data.raise_for_status()  # Raise an exception for bad status codes
        file_path = os.path.join(category_dir, f'{sound_name}_{sound_id}.mp3')
        with open(file_path, 'wb') as f:
            f.write(audio_data.content)
        print(f'Successfully downloaded: {sound_name}_{sound_id}.mp3')
        
        # Add metadata to CSV file
        with open(metadata_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Unpack metadata tuple
            tempo, description, num_downloads, duration, license_url, username = metadata

            # Write metadata to the CSV
            writer.writerow([
                f'{sound_name}_{sound_id}.mp3',  # File name
                category,                        # Category
                tempo,                           # Tempo
                ', '.join(tags),                 # Tags
                description,                     # Description
                num_downloads,                   # Number of downloads
                duration,                        # Duration
                license_url,                     # License
                username                         # Username
            ])
            
    except requests.exceptions.RequestException as e:
        print(f"Error downloading sound: {e}")


if __name__ == '__main__':
    all_sounds = []

    for category, subfolder in CATEGORY_MAPPING.items():
        print(f"Searching for sounds in category: {category}")
        sounds = search_sounds(category, num_results=10)  # Search without license filter
        all_sounds.extend(sounds)

    # Remove duplicate sounds based on their ID (if any)
    unique_sounds = {sound['id']: sound for sound in all_sounds}.values()

    # Process and download each sound
    for sound in unique_sounds:
        sound_id = sound['id']
        sound_name = sound['name']
        tags = sound['tags']
        preview_url = sound['previews']['preview-hq-mp3']  # Use the high-quality MP3 preview URL
        metadata = fetch_metadata(sound_id)
        download_sound(sound_id, sound_name, tags, preview_url, metadata)
