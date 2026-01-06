
from sklearn.decomposition import PCA
import lyricsgenius
import re
import os
import pickle
from dotenv import load_dotenv

load_dotenv()

GENIUS_ID = os.getenv("GENIUS_ACCESS_TOKEN")

if not GENIUS_ID:
    raise RuntimeError("GENIUS_ACCESS_TOKEN not found. Make sure .env is loaded properly.")

genius = lyricsgenius.Genius(GENIUS_ID, remove_section_headers = True)

def get_song(song_title, artist_title):

    
    return genius.search_song(song_title, artist_title)


def get_song_cached(song_name, artist_name, cache_dir = 'lyrics_cache'):

    os.makedirs(cache_dir, exist_ok = True)

    safe_name = f"{song_name}__{artist_name}".replace(" ", "_").lower()
    cache_path = os.path.join(cache_dir, f"{safe_name}.pkl")

    if os.path.exists(cache_path):
        print('Song found!')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    song = genius.search_song(song_name, artist_name)
    if song:
        print('Song not cached - caching song')
        with open(cache_path, 'wb') as f:
            pickle.dump(song, f)

    return song


def get_lyrics(song_title, artist_title):
    song = genius.search_song(song_title, artist_title)
    return song.lyrics if song else None


### Cleaning


def clean_lyrics(text):
    
    text = re.sub(r'^\d+\s+Contributors.*$', '', text, flags=re.MULTILINE)
    text = text.replace('—', ' — ').replace('–', ' – ')
    text = re.sub(r'\[(.*?)\]', '', text)
    text = re.sub(r'[^a-zA-Z\s\-]', '', text, flags=re.DOTALL)
    text = '\n'.join(re.sub(r'\s+', ' ', line).strip() for line in text.split('\n'))
    

    return text


def filter_and_split_lines(text, max_length=200):
    lines = text.split('\n')
    filtered_lines = [line for line in lines if (len(line.strip()) <= max_length and line.strip() != '')]
    return filtered_lines

def remove_duplicates(lines):
    return list(dict.fromkeys(lines))

def lyrics_preprocessing(song):
    text = clean_lyrics(song)
    lines = filter_and_split_lines(text)
    
    return lines

