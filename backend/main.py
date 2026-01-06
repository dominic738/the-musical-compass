import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import generative_utils as cu
import embedding_utils as eu
import genius_utils as gu
import spotify_utils as su
from concurrent.futures import ThreadPoolExecutor, as_completed
import uvicorn
import numpy as np

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


def safe_generate_dynamic_axis_phrases(word, timeout=90):
    """
    Runs generate_dynamic_axis_phrases in a thread with timeout.
    Returns empty array if the call fails or times out.
    90s timeout to handle phrase generation + embedding (~5s + ~3s per word).
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(cu.generate_dynamic_axis_phrases, word)
            result = future.result(timeout=timeout)
            return result if result is not None else np.array([])
    except TimeoutError:
        print(f"Timeout for word '{word}'", flush=True)
        return np.array([])
    except Exception as e:
        print(f"LLM failed for '{word}': {e}", flush=True)
        return np.array([])


@app.post("/generate-axes")
async def generate_axes(request: Request):
    data = await request.json()
    x_pos = data.get("x_pos", "")
    x_neg = data.get("x_neg", "")
    y_pos = data.get("y_pos", "")
    y_neg = data.get("y_neg", "")

    print("Received request:", data, flush=True)

    # Generate phrases for all 4 words in parallel
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
                embeddings = future.result()
                results[word] = embeddings
                print(f"Finished embeddings for '{word}' -> {len(embeddings)} phrase embeddings", flush=True)
            except Exception as e:
                print(f"Error processing '{word}': {e}", flush=True)
                results[word] = np.array([])

    # Extract phrase embedding arrays
    print('Extracting phrase embeddings', flush=True)
    x_pos_embeddings = results.get(x_pos, np.array([]))
    x_neg_embeddings = results.get(x_neg, np.array([]))
    y_pos_embeddings = results.get(y_pos, np.array([]))
    y_neg_embeddings = results.get(y_neg, np.array([]))

    # Validate we have embeddings for all words
    if len(x_pos_embeddings) == 0 or len(x_neg_embeddings) == 0 or len(y_pos_embeddings) == 0 or len(y_neg_embeddings) == 0:
        missing = [word for word, embs in results.items() if len(embs) == 0]
        print(f"Missing embeddings for words: {missing}. Aborting axis computation.", flush=True)
        raise HTTPException(status_code=400, detail=f"No phrase embeddings generated for: {missing}")

    # Compute axes using OpenAI embeddings + SVM
    print('Computing Axes', flush=True)
    try:
        axis_x = eu.compute_axis_svm(x_pos_embeddings, x_neg_embeddings)
        axis_y = eu.compute_axis_svm(y_pos_embeddings, y_neg_embeddings)
    except Exception as e:
        print(f"Error computing axes: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Axis computation failed: {str(e)}")

    print("Returning axes", flush=True)
    return {"axis_x": axis_x.tolist(), "axis_y": axis_y.tolist()}


@app.post("/get-playlist-tracks")
async def get_playlist(request: Request):
    data = await request.json()
    
    playlist_url = data.get('playlist_url')
    playlist = su.get_playlist(playlist_url)

    print(f"Found {len(playlist)} tracks in playlist", flush=True)
    tracks = [{"title": title, "artist": artist} for title, artist in playlist.items()]
    return {"tracks": tracks}


@app.post("/embed-song")
async def embed_song(request: Request):
    """
    Embed a single song using the 'Smart' logic:
    1. Checks Pinecone DB first (Fast)
    2. Only fetches from Genius if DB miss (Slow)
    """
    data = await request.json()
    title = data.get('title')
    artist = data.get('artist')

    if not title or not artist:
        return {'error': "Missing title or artist"}

    print(f'Embedding song: {title} — {artist}', flush=True)

    try:
        # 1. Use the new Smart function (Assumed to be in your 'eu' module)
        # This handles the DB check AND the Genius fallback internally.
        weighted_embedding = eu.get_weighted_embedding(title, artist)
        
        if weighted_embedding is None:
             return {'error': 'Song not found or no lyrics available.'}

        print(f'✓ Embedding ready for: {title}', flush=True)

        return {
            "embedding": weighted_embedding.tolist()
        }

    except Exception as e:
        print(f"Error processing song: {e}", flush=True)
        return {'error': f"Server error: {str(e)}"}




@app.post("/reset-pinecone")
async def reset_pinecone(request: Request):
    """
    DANGER: Delete all vectors from both Pinecone indexes.
    Use with caution - this wipes all cached data.
    """
    data = await request.json()
    confirm = data.get('confirm', '')
    
    if confirm != 'DELETE_ALL':
        raise HTTPException(
            status_code=400, 
            detail="Must send {\"confirm\": \"DELETE_ALL\"} to reset database"
        )
    
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        song_index = pc.Index("song-embeddings")
        phrase_index = pc.Index("phrase-embeddings")
        
        results = []
        
        # Try to delete all vectors from song-embeddings
        try:
            print("Deleting all vectors from song-embeddings...", flush=True)
            song_index.delete(delete_all=True, namespace="")
            results.append("song-embeddings: deleted")
        except Exception as e:
            if "not found" in str(e).lower():
                results.append("song-embeddings: already empty")
            else:
                raise e
        
        # Try to delete all vectors from phrase-embeddings
        try:
            print("Deleting all vectors from phrase-embeddings...", flush=True)
            phrase_index.delete(delete_all=True, namespace="")
            results.append("phrase-embeddings: deleted")
        except Exception as e:
            if "not found" in str(e).lower():
                results.append("phrase-embeddings: already empty")
            else:
                raise e
        
        print("✓ Pinecone reset complete", flush=True)
        
        return {
            "status": "success",
            "message": "Reset complete",
            "details": results
        }
        
    except Exception as e:
        print(f"Error resetting Pinecone: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)