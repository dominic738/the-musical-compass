import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import re
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

if not CLIENT_ID and not CLIENT_SECRET:
    raise RuntimeError("CLIENT_ID and/or CLIENT_SECRET not found. Make sure .env is loaded properly.")

auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)

sp = spotipy.Spotify(auth_manager=auth_manager)




def clean_track_name(name):
    # Remove things like: " - 2004 Digital Remaster", "(Remastered 2017)", etc.
    name = re.sub(
        r'\s*[\(\[-–—]?[^\)\]\-–—]*?(?:\d{4}\s*)?(Digital\s*)?Remaster(ed)?|Live|Bonus[^\)\]\-–—]*?[\)\]-–—]?',
        '',
        name,
        flags=re.IGNORECASE
    )
    # Remove any trailing punctuation or whitespace
    name = re.sub(r"[-–—'\s]+$", '', name)
    return name.strip()


def get_playlist(playlist_url):
    results = sp.playlist_items(playlist_url)
    song_d = {}

    
    while results:
        for item in results['items']:
            track = item.get('track')
            if not track:
                continue

            track_name = track['name']
            track_name = clean_track_name(track_name)
            
            artist_name = track['artists'][0]['name']
            song_d[track_name] = artist_name

        if results['next']:
            results = sp.next(results)
        else:
            break

    return song_d