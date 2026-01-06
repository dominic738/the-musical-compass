import genius_utils as gu
import numpy as np
import re
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDClassifier
from openai import OpenAI
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
song_index = pc.Index("song-embeddings")
phrase_index = pc.Index("phrase-embeddings")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def get_openai_embeddings(texts, batch_size=100):
    """
    Get embeddings from OpenAI API in batches.
    Returns numpy array of embeddings.
    """
    if isinstance(texts, str):
        texts = [texts]
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai_client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings)


def filter_to_centroid(phrases, embeddings, top_k=100):
    """Filter embeddings to top_k closest to centroid"""
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    sims = cosine_similarity(embeddings, centroid).flatten()
    top_indices = np.argsort(sims)[-top_k:]
    
    filtered_embeddings = np.array([embeddings[i] for i in top_indices])
    return filtered_embeddings


def sigmoid_scaled(x, scale=30):
    """Custom sigmoid scaler for distribution"""
    return 2 * ((1 / (1 + np.exp(-x * scale))) - 0.5)


def count_lines(lines):
    """Count duplicate lines"""
    return Counter(lines)


def normalize(v):
    """Normalize vector to unit length"""
    return v / np.linalg.norm(v)


def compute_axis_svm(axis1_embeddings, axis2_embeddings, filter_emb=True):
    """
    Construct semantic axis from two sets of phrase embeddings using SVM.
    Now accepts numpy arrays of embeddings directly.
    """
    # Convert to numpy arrays if needed
    if not isinstance(axis1_embeddings, np.ndarray):
        axis1_embeddings = np.array(axis1_embeddings)
    if not isinstance(axis2_embeddings, np.ndarray):
        axis2_embeddings = np.array(axis2_embeddings)
    
    print(f"Computing axis with {len(axis1_embeddings)} vs {len(axis2_embeddings)} embeddings")
    
    if filter_emb:
        axis1_embeddings = filter_to_centroid([], axis1_embeddings)
        axis2_embeddings = filter_to_centroid([], axis2_embeddings)
    
    if len(axis1_embeddings) == 0 or len(axis2_embeddings) == 0:
        raise ValueError("Not enough embeddings to build axis.")
    
    # Train SVM
    X = np.vstack([axis1_embeddings, axis2_embeddings])
    y = np.array([1] * len(axis1_embeddings) + [0] * len(axis2_embeddings))
    
    clf = SGDClassifier(loss="hinge", max_iter=5000, n_jobs=-1)
    clf.fit(X, y)
    axis = clf.coef_.flatten()
    axis = axis / np.linalg.norm(axis)
    
    return axis

def get_weighted_embedding(title, artist):
    """
    1. Check Pinecone first (using Title + Artist).
    2. If found -> Return embedding (Skip Genius entirely!).
    3. If NOT found -> Fetch lyrics from Genius, calculate embedding, and save.
    """
    
    # 1. Create the ID exactly how you did in get_weighted_embedding
    song_id = f"{title}__{artist}".replace(" ", "_").lower()
    song_id = re.sub(r'[^a-zA-Z0-9_-]', '', song_id)

    # 2. CHECK PINECONE FIRST
    try:
        # We assume 'song_index' is your defined Pinecone index variable
        result = song_index.fetch(ids=[song_id])
        if song_id in result['vectors']:
            print(f"✓ Found cached embedding in Pinecone for: {title}")
            # Return the vector immediately
            return np.array(result['vectors'][song_id]['values'])
    except Exception as e:
        print(f"Pinecone check failed (continuing to Genius): {e}")

    # 3. IF WE ARE HERE, IT WAS NOT IN PINECONE
    # Now we actually need to do the heavy lifting (Genius -> Lyrics -> OpenAI)
    print(f"Song not in DB. Fetching from Genius: {title}")
    
    # Use your existing get_song (or the cached version if running locally)
    song = gu.get_song(title, artist)
    
    if not song:
        print(f"Could not find song on Genius: {title}")
        return None

    # 4. COMPUTE EMBEDDING
    # (This logic is copied from your get_weighted_embedding)
    cleaned_lines = gu.lyrics_preprocessing(song.lyrics)
    line_counts = count_lines(cleaned_lines) # Ensure this function is defined in your code
    unique_lines = list(line_counts.keys())
    counts = np.array([line_counts[line] for line in unique_lines])
    
    if len(unique_lines) == 0 or np.sum(counts) == 0:
        return None
    
    embeddings = get_openai_embeddings(unique_lines) # Ensure this function is defined
    weighted_embedding = np.average(embeddings, axis=0, weights=counts)

    # 5. SAVE TO PINECONE (So next time we skip step 3 & 4)
    try:
        song_index.upsert(vectors=[{
            "id": song_id,
            "values": weighted_embedding.tolist(),
            "metadata": {
                "title": song.title,
                "artist": song.artist,
                # Optional: Save preview if you want, but strictly not needed for math
                "lyrics_preview": song.lyrics[:200] if song.lyrics else ""
            }
        }])
        print(f"✓ Saved new embedding to Pinecone: {title}")
    except Exception as e:
        print(f"Pinecone upsert error: {e}")

    return weighted_embedding


def score_song_weighted(song, axis):
    """
    Score a song by projecting its weighted embedding onto an axis.
    """
    weighted_embedding = get_weighted_embedding(song)
    
    if weighted_embedding is None:
        return None
    
    # Project onto axis
    projection = np.dot(weighted_embedding, axis)
    sigmoid_score = sigmoid_scaled(projection)
    
    return sigmoid_score


def score_playlist(playlist, axis):
    """
    Score a playlist by projecting embeddings onto a semantic axis.
    Optimized to use Pinecone cache first.
    """
    score_d = {}
    
    # Iterate through the dictionary (Title -> Artist)
    for song_title, artist_name in playlist.items():
        try:
            # 1. GET VECTOR (Checks Pinecone -> Then Genius -> Then OpenAI)
            # This uses the helper function we wrote in the previous step
            embedding = get_weighted_embedding(song_title, artist_name)
            
            if embedding is None:
                print(f"Skipping '{song_title}' — could not generate embedding.")
                continue
            
            # 2. CALCULATE SCORE
            # Since we have the raw vector now, we just do the Dot Product
            score = np.dot(embedding, axis)
            
            # 3. STORE RESULT
            score_d[f"{song_title} — {artist_name}"] = score
            
        except Exception as e:
            print(f"Error scoring '{song_title}': {e}")
            continue
            
    return score_d


def score_playlist_2D(playlist, x_axis, y_axis):
    """
    Score a playlist on 2D axes (x and y).
    Uses cached embeddings from Pinecone when available.
    """
    score_d = {}
    
    # Iterate through the dictionary (Title -> Artist)
    for song_title, artist_name in playlist.items():
        try:
            # 1. GET VECTOR (Checks Pinecone -> Then Genius -> Then OpenAI)
            # We only need to fetch this ONCE for both axes
            embedding = get_weighted_embedding(song_title, artist_name)
            
            if embedding is None:
                print(f"Skipping '{song_title}' — could not generate embedding.")
                continue
            
            # 2. CALCULATE BOTH SCORES
            # Use the same embedding for both dot products
            x_score = np.dot(embedding, x_axis)
            y_score = np.dot(embedding, y_axis)
            
            # 3. STORE RESULT AS TUPLE
            score_d[f"{song_title} — {artist_name}"] = (x_score, y_score)
        
        except Exception as e:
            print(f"Error scoring '{song_title}': {e}")
            continue
    
    return score_d


def create_score_group(scores_dict, color, label):
    """Helper function to create score groups"""
    return (scores_dict, color, label)