import lyrics_utils as lu
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

def get_weighted_embedding_from_lyrics(title, artist, lyrics_text):
    """
    Embed lyrics text only.
    Backend NEVER fetches lyrics.
    """

    if "<html" in lyrics_text.lower():
        raise ValueError("Received HTML instead of lyrics text")

    # 1. Create stable ID
    song_id = f"{title}__{artist}".replace(" ", "_").lower()
    song_id = re.sub(r'[^a-zA-Z0-9_-]', '', song_id)

    # 2. Check Pinecone cache
    try:
        result = song_index.fetch(ids=[song_id])
        if song_id in result["vectors"]:
            print(f"✓ Found cached embedding for: {title}")
            return np.array(result["vectors"][song_id]["values"])
    except Exception:
        pass

    # 3. CLEAN + PREPROCESS PROVIDED LYRICS
    cleaned_lines = lu.lyrics_preprocessing(lyrics_text)
    line_counts = count_lines(cleaned_lines)

    unique_lines = list(line_counts.keys())
    counts = np.array([line_counts[line] for line in unique_lines])

    if len(unique_lines) == 0 or np.sum(counts) == 0:
        return None

    # 4. EMBED
    embeddings = get_openai_embeddings(unique_lines)
    weighted_embedding = np.average(
        embeddings, axis=0, weights=counts
    )

    # 5. SAVE TO PINECONE
    song_index.upsert([{
        "id": song_id,
        "values": weighted_embedding.tolist(),
        "metadata": {
            "title": title,
            "artist": artist
        }
    }])

    return weighted_embedding

