import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import generative_utils as cu
import embedding_utils as eu
import lyrics_utils as lu
import spotify_utils as su
from concurrent.futures import ThreadPoolExecutor, wait
import uvicorn
import numpy as np
from pinecone import Pinecone


app = FastAPI()

ALLOWED_ORIGINS = [
    "https://mymusicalcompass.com",
    "https://www.mymusicalcompass.com",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "null",  # local index.html opened directly via file:// sends Origin: null
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Key"],
)

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

def require_admin_key(request: Request):
    if not ADMIN_API_KEY or request.headers.get("X-Admin-Key") != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Not authorized")


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
song_index = pc.Index("song-embeddings")

# Total time allowed for phrase generation + embedding across all axis words
AXIS_TIMEOUT_SECONDS = 90


class AxesRequest(BaseModel):
    x_pos: str = Field(min_length=1, max_length=50)
    x_neg: str = Field(min_length=1, max_length=50)
    y_pos: str = Field(min_length=1, max_length=50)
    y_neg: str = Field(min_length=1, max_length=50)


class PlaylistRequest(BaseModel):
    playlist_url: str = Field(min_length=1, max_length=500)


class SongRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    artist: str = Field(min_length=1, max_length=200)


class LyricsRequest(SongRequest):
    lyrics: str = Field(min_length=1, max_length=20000)


class ResetRequest(BaseModel):
    confirm: str = ""

print("FastAPI app starting...", flush=True)

@app.get("/")
def read_root():
    return {"hello": "world"}

@app.post("/test-post")
async def test_post(data: dict):
    print("POST received!", flush=True)
    return {"ok": True}


@app.post("/generate-axes")
def generate_axes(body: AxesRequest):
    x_pos, x_neg = body.x_pos.strip(), body.x_neg.strip()
    y_pos, y_neg = body.y_pos.strip(), body.y_neg.strip()

    print("Received request:", body.model_dump(), flush=True)

    if not all([x_pos, x_neg, y_pos, y_neg]):
        raise HTTPException(status_code=400, detail="Please fill in all four axis words.")
    if x_pos.lower() == x_neg.lower() or y_pos.lower() == y_neg.lower():
        raise HTTPException(status_code=400, detail="Each axis needs two different words.")

    # Phrases are cached per lowercase word, so only generate each distinct word once
    words = {w.lower(): w for w in [x_pos, x_neg, y_pos, y_neg]}

    # Generate phrases for all words in parallel. Not using a `with` block: its
    # exit waits for every thread, which would defeat the timeout.
    executor = ThreadPoolExecutor(max_workers=len(words))
    futures = {executor.submit(cu.generate_dynamic_axis_phrases, w): key for key, w in words.items()}
    done, not_done = wait(futures, timeout=AXIS_TIMEOUT_SECONDS)
    executor.shutdown(wait=False, cancel_futures=True)

    results = {}
    for future in done:
        key = futures[future]
        try:
            embeddings = future.result()
            results[key] = embeddings if embeddings is not None else np.array([])
            print(f"Finished embeddings for '{key}' -> {len(results[key])} phrase embeddings", flush=True)
        except Exception as e:
            print(f"LLM failed for '{key}': {e}", flush=True)
    for future in not_done:
        print(f"Timeout for word '{futures[future]}'", flush=True)

    missing = [w for key, w in words.items() if len(results.get(key, [])) == 0]
    if missing:
        print(f"Missing embeddings for words: {missing}. Aborting axis computation.", flush=True)
        raise HTTPException(status_code=400, detail=f"No phrase embeddings generated for: {missing}")

    # Compute axes using OpenAI embeddings + SVM
    print('Computing Axes', flush=True)
    try:
        axis_x, offset_x = eu.compute_axis_svm(results[x_pos.lower()], results[x_neg.lower()])
        axis_y, offset_y = eu.compute_axis_svm(results[y_pos.lower()], results[y_neg.lower()])
    except Exception as e:
        print(f"Error computing axes: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Axis computation failed: {str(e)}")

    print("Returning axes", flush=True)
    return {
        "axis_x": axis_x.tolist(),
        "axis_y": axis_y.tolist(),
        "offset_x": offset_x,
        "offset_y": offset_y,
    }


@app.post("/get-playlist-tracks")
def get_playlist(body: PlaylistRequest):
    try:
        playlist = su.get_playlist(body.playlist_url)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    print(f"Found {len(playlist)} tracks in playlist", flush=True)
    tracks = [{"title": title, "artist": artist} for title, artist in playlist.items()]
    return {"tracks": tracks}


@app.post("/check-song")
def check_song(body: SongRequest):
    song_id = eu.make_song_id(body.title, body.artist)

    try:
        result = song_index.fetch(ids=[song_id])
        if song_id in result["vectors"]:
            return {
                "found": True,
                "embedding": result["vectors"][song_id]["values"]
            }
    except Exception as e:
        print(f"Pinecone fetch error for '{song_id}': {e}", flush=True)

    return {"found": False}


@app.post("/embed-lyrics")
def embed_song(body: LyricsRequest):
    title, artist, lyrics = body.title.strip(), body.artist.strip(), body.lyrics

    print(f'Embedding song: {title} — {artist}', flush=True)

    try:
        weighted_embedding = eu.get_weighted_embedding_from_lyrics(title, artist, lyrics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error processing song: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    if weighted_embedding is None:
        raise HTTPException(status_code=400, detail="No usable lyrics found in the pasted text.")

    print(f'✓ Embedding ready for: {title}', flush=True)
    return {"embedding": weighted_embedding.tolist()}


@app.post("/reset-pinecone")
def reset_pinecone(request: Request, body: ResetRequest):
    """
    DANGER: Delete all vectors from both Pinecone indexes.
    Use with caution - this wipes all cached data.
    Requires the X-Admin-Key header to match the ADMIN_API_KEY env var.
    """
    require_admin_key(request)

    if body.confirm != 'DELETE_ALL':
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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)