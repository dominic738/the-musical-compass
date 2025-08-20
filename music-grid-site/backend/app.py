import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import generative_utils_lite as cu
import embedding_utils as eu
import genius_utils as gu
import spotify_utils as su
from collections import Counter


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-axes")
async def generate_axes(request: Request):
    data = await request.json()
    x_pos = data["x_pos"]
    x_neg = data["x_neg"]
    y_pos = data["y_pos"]
    y_neg = data["y_neg"]


    print('Starting phrase generation')

    x_pos_phrases = cu.generate_dynamic_axis_phrases(x_pos)
    x_neg_phrases = cu.generate_dynamic_axis_phrases(x_neg)
    y_pos_phrases = cu.generate_dynamic_axis_phrases(y_pos)
    y_neg_phrases = cu.generate_dynamic_axis_phrases(y_neg)

    print('Done!')

    print('Starting axis generation')


    axis_x = eu.compute_axis_svm(x_pos_phrases, x_neg_phrases)
    axis_y = eu.compute_axis_svm(y_pos_phrases, y_neg_phrases)

    print('Done!')

    return {
        "axis_x": axis_x.tolist(),
        "axis_y": axis_y.tolist()
    }

@app.post("/get-playlist-tracks")
async def get_playlist(request: Request):
    data = await request.json()
    
    playlist_url = data.get('playlist_url')
    playlist = su.get_playlist(playlist_url)

    print(playlist)
    tracks = [{"title": title, "artist": artist} for title, artist in playlist.items()]
    return {"tracks" : tracks}



@app.post("/embed-song")
async def embed_song(request: Request):
    data = await request.json()
    title = data['title']
    artist = data['artist']

    print('Starting embedding')

    try:
        song = gu.get_song_cached(title, artist)
    except Exception as e:
        print(f"Genius API failed: {e}")
        return {'error': "Genius API timeout or error"}
    

    if not song or not hasattr(song, 'lyrics') or not song.lyrics:
        return {'error' : 'No lyrics found.'}
    

    
    weighted_embedding = eu.get_weighted_embedding(song)

    if weighted_embedding is None:
        return {'error': 'Lyrics found but no valid tokens for embedding.'}

    print('Here it is: ', weighted_embedding)

    return {
        "embedding": weighted_embedding.tolist()
    }
    