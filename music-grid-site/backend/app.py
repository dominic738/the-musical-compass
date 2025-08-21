import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import generative_utils_lite as cu
import embedding_utils as eu
import genius_utils as gu
import spotify_utils as su
from collections import Counter
import uvicorn
from concurrent.futures import ThreadPoolExecutor, as_completed




app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("FastAPI app starting...", flush=True)

@app.get("/")
def read_root():
    return {"hello": "world"}

@app.post("/test-post")
async def test_post(data: dict):
    print("POST received!", flush=True)
    return {"ok": True}


def safe_generate_dynamic_axis_phrases(word, timeout=10):
    """
    Runs generate_dynamic_axis_phrases in a thread with timeout.
    Returns [] if the call fails or times out.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(cu.generate_dynamic_axis_phrases, word)
            return future.result(timeout=timeout)
    except TimeoutError:
        print(f"Timeout for word '{word}'", flush=True)
        return []
    except Exception as e:
        print(f"LLM failed for '{word}': {e}", flush=True)
        return []

@app.post("/generate-axes")
async def generate_axes(request: Request):
    data = await request.json()
    x_pos = data.get("x_pos", "")
    x_neg = data.get("x_neg", "")
    y_pos = data.get("y_pos", "")
    y_neg = data.get("y_neg", "")

    print("Received request:", data, flush=True)

    # Prepare threads for all words
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(safe_generate_dynamic_axis_phrases, word): word
            for word in [x_pos, x_neg, y_pos, y_neg]
        }

        # Collect results as they finish
        for future in as_completed(futures):
            word = futures[future]
            try:
                phrases = future.result()
                results[word] = phrases
                print(f"Finished phrases for '{word}' -> {len(phrases)} phrases", flush=True)
            except Exception as e:
                print(f"Error processing '{word}': {e}", flush=True)
                results[word] = []

    # Extract phrase lists

    print('Extracting phrase lists', flush=True)
    x_pos_phrases = results.get(x_pos, [])
    x_neg_phrases = results.get(x_neg, [])
    y_pos_phrases = results.get(y_pos, [])
    y_neg_phrases = results.get(y_neg, [])

    # Compute axes safely

    if not x_pos_phrases or not x_neg_phrases or not y_pos_phrases or not y_neg_phrases:
        missing = [word for word, phrases in results.items() if not phrases]
        print(f"Missing phrases for words: {missing}. Aborting axis computation.", flush=True)
        raise HTTPException(status_code=400, detail=f"No phrases generated for: {missing}")

    print('Computing Axes', flush=True)
    try:
        axis_x = eu.compute_axis_svm(x_pos_phrases, x_neg_phrases) if x_pos_phrases and x_neg_phrases else [0.0, 0.0]
        axis_y = eu.compute_axis_svm(y_pos_phrases, y_neg_phrases) if y_pos_phrases and y_neg_phrases else [0.0, 0.0]
    except Exception as e:
        print(f"Error computing axes: {e}", flush=True)
        axis_x, axis_y = [0.0, 0.0], [0.0, 0.0]

    print("Returning axes", flush=True)
    return {"axis_x": axis_x.tolist(), "axis_y": axis_y.tolist()}

"""
@app.post("/generate-axes")
async def generate_axes(request: Request):
    data = await request.json()
    x_pos = data["x_pos"]
    x_neg = data["x_neg"]
    y_pos = data["y_pos"]
    y_neg = data["y_neg"]


    print('Starting phrase generation')

    try:

        x_pos_phrases = cu.generate_dynamic_axis_phrases(x_pos)
        x_neg_phrases = cu.generate_dynamic_axis_phrases(x_neg)
        y_pos_phrases = cu.generate_dynamic_axis_phrases(y_pos)
        y_neg_phrases = cu.generate_dynamic_axis_phrases(y_neg)
    
    except Exception as e:
        print("Error generating phrases:", e, flush=True)

    print('Done!')

    print('Starting axis generation')


    axis_x = eu.compute_axis_svm(x_pos_phrases, x_neg_phrases)
    axis_y = eu.compute_axis_svm(y_pos_phrases, y_neg_phrases)

    print('Done!')

    return {
        "axis_x": axis_x.tolist(),
        "axis_y": axis_y.tolist()
    }

"""

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
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Railway gives you a PORT, fallback=8000 for local
    uvicorn.run(app, host="0.0.0.0", port=port)