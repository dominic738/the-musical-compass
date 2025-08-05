import genius_utils as gu
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
import json
from embedding_utils import filter_to_centroid, sigmoid_scaled, count_lines, normalize, compute_axis_svm, score_song_weighted, score_playlist, score_playlist_2D, create_score_group




def export_scores_to_json(score_dict, color="dodgerblue", filename="frontend/songs.json"):
    output = []
    for title, (x, y) in score_dict.items():
        output.append({
            "title": title,
            "x": x,
            "y": y,
            "color": color
        })
    
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Exported {len(output)} songs to {filename}")