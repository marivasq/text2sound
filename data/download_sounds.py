import requests

# Replace with your own API key
API_KEY = 'your_api_key'
BASE_URL = 'https://freesound.org/apiv2/'

def search_sounds(query, num_results=10):
    params = {
        'query': query,
        'fields': 'id,name,tags,previews',
        'token': API_KEY,
    }
    response = requests.get(BASE_URL + 'sounds/search/', params=params)
    if response.status_code == 200:
        results = response.json()
        return results['results'][:num_results]
    return []

def download_sound(sound_id):
    download_url = f'https://freesound.org/data/previews/{sound_id[:3]}/{sound_id[3:6]}/{sound_id}.mp3'
    audio_data = requests.get(download_url)
    with open(f'sound_{sound_id}.mp3', 'wb') as f:
        f.write(audio_data.content)
    print(f'Downloaded sound_{sound_id}.mp3')

sounds = search_sounds('drum kick')
for sound in sounds:
    download_sound(sound['id'])
