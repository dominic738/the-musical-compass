
from sklearn.decomposition import PCA
import lyricsgenius
import re
import os
import pickle
from dotenv import load_dotenv
import requests


### Cleaning


def clean_lyrics(text):
    
    text = re.sub(r'^\d+\s+Contributors.*$', '', text, flags=re.MULTILINE)
    text = text.replace('—', ' — ').replace('–', ' – ')
    text = re.sub(r'\[(.*?)\]', '', text)
    text = text.replace('’', "'").replace('‘', "'")
    # Keep letters from any script (accents, CJK, Cyrillic...) plus apostrophes and hyphens
    text = re.sub(r"[^\w\s'\-]|[\d_]", '', text)
    text = '\n'.join(re.sub(r'\s+', ' ', line).strip() for line in text.split('\n'))

    return text


def filter_and_split_lines(text, max_length=200):
    lines = text.split('\n')
    filtered_lines = [line for line in lines if (len(line.strip()) <= max_length and line.strip() != '')]
    return filtered_lines

def lyrics_preprocessing(lyrics_test: str):
    text = clean_lyrics(lyrics_test)
    lines = filter_and_split_lines(text)
    
    return lines


