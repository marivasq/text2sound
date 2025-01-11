import os
import requests
import csv
import webbrowser

# Set up the folder where the sounds will be saved
BASE_SAVE_DIR = os.path.join(os.getcwd(), 'dataset', 'raw')

# Ensure the directory exists
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# FreeSound API settings
BASE_URL = 'https://freesound.org/apiv2/'

# Replace these with your credentials
CLIENT_ID = ''
CLIENT_SECRET = ''
REDIRECT_URI = "https://freesound.org/home/app_permissions/permission_granted/"
TOKEN_URL = "https://freesound.org/apiv2/oauth2/access_token/"
AUTH_URL = "https://freesound.org/apiv2/oauth2/authorize/"
DOWNLOAD_URL_TEMPLATE = "https://freesound.org/apiv2/sounds/{sound_id}/download/"

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
metadata_file = os.path.join(BASE_SAVE_DIR, 'raw_metadata.csv')
with open(metadata_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write headers for the metadata CSV file
    writer.writerow(['filename', 'sound id', 'category', 'tempo', 'tags', 'description', 'num_downloads', 'duration', 'license', 'username'])

def authorize_user():
    """
    Directs the user to the authorization URL to log in and authorize the app.
    """
    
    auth_url = f"{AUTH_URL}?client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}"
    print(f"Opening the authorization URL. Log in and authorize the app: {auth_url}")
    webbrowser.open(auth_url)
    auth_code = input("After authorizing, enter the authorization code here: ")
    return auth_code

def get_access_token(auth_code):
    # Prepare the payload
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': 'authorization_code',
        'code': auth_code,
    }

    # Debug print to check the payload
    print(f"Sending token request with payload: {payload}")

    # Make the POST request to the access token endpoint
    response = requests.post(TOKEN_URL, data=payload)
    
    # Debug print for the response
    print(f"Response status code: {response.status_code}")
    print(f"Response text: {response.text}")

    # Handle response
    if response.status_code == 200:
        token_data = response.json()
        print("Access Token:", token_data["access_token"])
        print("Refresh Token:", token_data.get("refresh_token", "N/A"))
        print("Token Scope:", token_data["scope"])
        print("Expires In:", token_data["expires_in"])
        return token_data
    else:
        raise Exception(f"Failed to obtain access token: {response.text}")

def fetch_metadata(sound_id):
    """
    Collects the metadata for a given sound ID from the Freesound API.

    Args:
        sound_id (int or str): The ID of the sound to fetch metadata for.
        
    Returns:
        tuple: A tuple containing the following metadata:
            - tempo (str): The tempo (BPM) of the sound or "N/A" if not available.
            - description (str): A brief description of the sound or "N/A" if not available.
            - num_downloads (int or str): The number of times the sound has been downloaded, or "N/A" if not available.
            - duration (float or str): The duration of the sound in seconds, or "N/A" if not available.
            - license_url (str): The URL of the license for the sound, or "N/A" if not available.
            - username (str): The username of the person who uploaded the sound, or "N/A" if not available.
    """
    url = f'{BASE_URL}sounds/{sound_id}/'
    analysis_url = f"{BASE_URL}sounds/{sound_id}/analysis/"
    headers = {
        "Authorization": f"Token {CLIENT_SECRET}"
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
    """
    Collects (int) num_results sounds based on the query which denotes the category.

    Returns:
        array: An array of dictionaries where each one coressponds to a unique sound collected.
    """

    params = {
        'query': query,
        'fields': 'id,name,tags,previews,license,num_downloads,download',
        'token': CLIENT_SECRET,
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


def download_sound(sound, metadata, access_token):
    """
    Downloads the original sound file and its coressponding metadata.
    """

    sound_id = sound['id']
    sound_name = sound['name']
    tags = sound['tags']

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
        url = DOWNLOAD_URL_TEMPLATE.format(sound_id=sound_id)
        headers = {
            "Authorization": f"Bearer {access_token}"
        }

        save_path = os.path.join(category_dir, sound_name)
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Sound downloaded successfully to {save_path}")
        else:
            print("Failed to download sound:", response.status_code, response.text)
        
        # Add metadata to CSV file
        with open(metadata_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            
            # Unpack metadata tuple
            tempo, description, num_downloads, duration, license_url, username = metadata

            # Write metadata to the CSV
            writer.writerow([
                sound_name,                      # File name
                sound_id,                        # Sound ID
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

    auth_code = authorize_user()
    token_response = get_access_token(auth_code)
    access_token = token_response["access_token"]

    # Process and download each sound
    for sound in unique_sounds:
        
        metadata = fetch_metadata(sound['id'])
        download_sound(sound, metadata, access_token)
