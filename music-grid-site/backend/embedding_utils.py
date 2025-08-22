
import genius_utils as gu
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

#MODEL = SentenceTransformer('all-mpnet-base-v2', device='cpu')
MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Filtering outliers for SVM

def filter_to_centroid(phrases, embeddings, top_k=100):
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    sims = cosine_similarity(embeddings, centroid).flatten()
    top_indices = np.argsort(sims)[-top_k:]

    filtered_embeddings = np.array([embeddings[i] for i in top_indices])
    
    return filtered_embeddings


# Custom Sigmoid Scaler for Distribution

def sigmoid_scaled(x, scale = 30):
    return 2 * ((1 / (1 + np.exp(-x * scale))) - 0.5)


# Count duplicate lines

def count_lines(lines):
    return Counter(lines)



# Normalize vector to unit length

def normalize(v):
    return v / np.linalg.norm(v)
    


# Construct semantic axis from two sets of phrases

def compute_axis_svm(axis1_phrases, axis2_phrases, model = MODEL, filter_emb = True):
    

    axis1_vecs = model.encode(axis1_phrases, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)
    axis2_vecs = model.encode(axis2_phrases, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)

    if filter_emb:
        axis1_vecs = filter_to_centroid(axis1_phrases, axis1_vecs.cpu().numpy())
        axis2_vecs = filter_to_centroid(axis2_phrases, axis2_vecs.cpu().numpy())
    else:
        axis1_vecs = axis1_vecs.cpu().numpy()
        axis2_vecs = axis2_vecs.cpu().numpy()

    if len(axis1_vecs) == 0 or len(axis2_vecs) == 0:
        raise ValueError("Not enough embeddings to build axis.")


    X = np.vstack([axis1_vecs, axis2_vecs])
    y = np.array([1] * len(axis1_vecs) + [0] * len(axis2_vecs))

    clf = LinearSVC(max_iter=5000, dual=False)
    clf.fit(X, y)

    axis = clf.coef_.flatten()
    return axis / np.linalg.norm(axis)



# Score a song by projecting embeddings onto axis with weighting

def score_song_weighted(song, axis, model = MODEL):
    cleaned_lines = gu.lyrics_preprocessing(song.lyrics)

    line_counts = count_lines(cleaned_lines)
    unique_lines = list(line_counts.keys())
    counts = np.array([line_counts[line] for line in unique_lines])

    embeddings = model.encode(unique_lines, batch_size = 128, normalize_embeddings = True)
    projections = np.dot(embeddings, axis)

    weighted_score = np.average(projections, weights = counts)
    
    return sigmoid_scaled(weighted_score)




def get_weighted_embedding(song, model = MODEL):
    cleaned_lines = gu.lyrics_preprocessing(song.lyrics)

    line_counts = count_lines(cleaned_lines)
    unique_lines = list(line_counts.keys())
    counts = np.array([line_counts[line] for line in unique_lines])

    embeddings = model.encode(unique_lines, normalize_embeddings = True)

    if len(embeddings) == 0 or np.sum(counts) == 0:
        return None

    weighted_embedding = np.average(embeddings, axis=0, weights=counts)
    return weighted_embedding



# Score a playlist using weighted song-level projections; use cached data for faster search

def score_playlist(playlist, axis, model = MODEL):
    score_d = {}
    for song_name in playlist.keys():
        try:
            artist_name = playlist[song_name]
            song = gu.get_song_cached(song_name, artist_name)
            if not song or not hasattr(song, 'lyrics') or not song.lyrics:
                print(f"Skipping '{song_name}' by {artist_name} — no lyrics found.")
                continue

            score = score_song_weighted(song, axis, model)
            score_d[f"{song_name} — {artist_name}"] = score

        except Exception as e:
            print(f"Error scoring '{song_name}' by {artist_name}: {e}")
            continue


    return score_d



def score_song_weighted(song, axis, model = MODEL):
    cleaned_lines = gu.lyrics_preprocessing(song.lyrics)

    line_counts = count_lines(cleaned_lines)
    unique_lines = list(line_counts.keys())
    counts = np.array([line_counts[line] for line in unique_lines])

    embeddings = model.encode(unique_lines, normalize_embeddings = True)
    projections = np.dot(embeddings, axis)

    weighted_score = np.average(projections, weights = counts)

    sigmoid_score = sigmoid_scaled(weighted_score)
    
    return sigmoid_score


def score_playlist_2D(playlist, x_axis, y_axis, model = MODEL):
    score_d = {}
    for song_name in playlist.keys():
        try:
            artist_name = playlist[song_name]
            song = gu.get_song_cached(song_name, artist_name)
            if not song or not hasattr(song, 'lyrics') or not song.lyrics:
                print(f"Skipping '{song_name}' by {artist_name} — no lyrics found.")
                continue

            x_score = score_song_weighted(song, x_axis, model)
            y_score = score_song_weighted(song, y_axis, model)
            score_d[f"{song_name} — {artist_name}"] = (x_score, y_score)

        except Exception as e:
            print(f"Error scoring '{song_name}' by {artist_name}: {e}")
            continue


    return score_d




def create_score_group(scores_dict, color, label):
    return (scores_dict, color, label)

