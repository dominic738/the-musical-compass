import re
import openai
import os
import uuid
import numpy as np
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
phrase_index = pc.Index("phrase-embeddings")

EMBEDDING_MODEL = "text-embedding-3-small"


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
        response = client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings)


MAX_CACHED_PHRASES = 500
FETCH_BATCH_SIZE = 100
# One vector per generated word lives in this namespace of the phrase index, so
# the words can be listed without scanning every phrase. Its values are the
# centroid of the word's phrases.
WORDS_NAMESPACE = "words"


def _field(obj, name):
    """Read a field from a Pinecone response object or plain dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def load_cached_phrases(word):
    """
    Load cached phrase embeddings from Pinecone.
    Returns numpy array of embeddings if a complete set is found, None otherwise.
    """
    try:
        # Pinecone doesn't support prefix search, so fetch the known ID range in
        # small batches (one huge GET can exceed URL length limits).
        ids_to_fetch = [f"{word.lower()}_{i}" for i in range(MAX_CACHED_PHRASES)]
        vectors = {}
        for i in range(0, len(ids_to_fetch), FETCH_BATCH_SIZE):
            result = phrase_index.fetch(ids=ids_to_fetch[i:i + FETCH_BATCH_SIZE])
            batch = _field(result, 'vectors') or {}
            if not batch:
                break  # IDs are contiguous, so an empty batch means we're past the end
            vectors.update(batch)

        if not vectors:
            return None

        # Only trust vectors from the same save as word_0, and only if all of them
        # made it in. Older entries without this metadata are accepted as-is.
        first_meta = _field(vectors.get(f"{word.lower()}_0"), 'metadata') or {}
        generation = first_meta.get('generation')
        expected = first_meta.get('total')
        if generation is not None:
            vectors = {
                vid: vec for vid, vec in vectors.items()
                if (_field(vec, 'metadata') or {}).get('generation') == generation
            }
            if expected is not None and len(vectors) != int(expected):
                print(f"Incomplete phrase cache for '{word}' ({len(vectors)}/{expected}), regenerating")
                return None

        embeddings = np.array([_field(vec, 'values') for vec in vectors.values()])
        print(f"✓ Loaded {len(embeddings)} cached phrase embeddings for '{word}' from Pinecone")
        # Words cached before the word list existed get added to it here
        register_word(word, embeddings, generation)
        return embeddings
    except Exception as e:
        print(f"Pinecone fetch error for '{word}': {e}")
    return None


def save_cached_phrases(word, phrases, embeddings):
    """
    Save phrase embeddings to Pinecone, with each phrase's text as metadata.
    Each embedding gets stored as a separate vector, tagged with a generation ID
    and the total count so partial or stale sets can be detected on load.
    """
    try:
        phrases = phrases[:MAX_CACHED_PHRASES]
        embeddings = embeddings[:MAX_CACHED_PHRASES]
        generation = uuid.uuid4().hex
        vectors = []
        for i, (phrase, embedding) in enumerate(zip(phrases, embeddings)):
            vectors.append({
                "id": f"{word.lower()}_{i}",
                "values": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                "metadata": {
                    "word": word.lower(),
                    "generation": generation,
                    "total": len(embeddings),
                    "text": phrase,
                }
            })
        
        # Upsert in batches of 100 (Pinecone limit)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            phrase_index.upsert(vectors=batch)
        
        print(f"✓ Cached {len(embeddings)} phrase embeddings for '{word}' in Pinecone")
        register_word(word, embeddings, generation)
    except Exception as e:
        print(f"Pinecone upsert error for '{word}': {e}")


def register_word(word, embeddings, generation=None):
    """Add or refresh the word's entry in the word list."""
    try:
        centroid = np.mean(embeddings, axis=0)
        metadata = {"word": word.lower(), "total": len(embeddings)}
        if generation is not None:
            metadata["generation"] = generation
        phrase_index.upsert(
            vectors=[{
                "id": word.lower(),
                "values": (centroid / np.linalg.norm(centroid)).tolist(),
                "metadata": metadata,
            }],
            namespace=WORDS_NAMESPACE,
        )
    except Exception as e:
        print(f"Couldn't register word '{word}': {e}")


def known_words(words):
    """Which of these (lowercase) words already have phrases cached."""
    try:
        result = phrase_index.fetch(ids=list(words), namespace=WORDS_NAMESPACE)
        return set((_field(result, 'vectors') or {}).keys())
    except Exception as e:
        print(f"Couldn't check known words: {e}")
        return set(words)  # unknown: don't claim anything is new


def list_words():
    """Every word that has phrases cached, sorted."""
    words = []
    for ids in phrase_index.list(namespace=WORDS_NAMESPACE):
        # Older SDKs yield plain ID strings, newer ones objects with an .id
        words.extend(i if isinstance(i, str) else _field(i, 'id') for i in ids)
    return sorted(w for w in words if w)


def count_words():
    stats = phrase_index.describe_index_stats()
    namespace = (_field(stats, 'namespaces') or {}).get(WORDS_NAMESPACE)
    return int(_field(namespace, 'vector_count') or 0) if namespace else 0


def load_phrase_texts(word):
    """
    The cached phrases for a word, most typical first (closest to the centroid,
    the same order the axis filter ranks them in). Returns None if the word has
    never been generated, and an empty list if it was cached before phrase text
    was saved.
    """
    key = word.lower()
    entry = (_field(phrase_index.fetch(ids=[key], namespace=WORDS_NAMESPACE), 'vectors') or {}).get(key)
    if entry is None:
        return None

    meta = _field(entry, 'metadata') or {}
    query_filter = {"word": {"$eq": key}}
    if meta.get("generation"):
        query_filter["generation"] = {"$eq": meta["generation"]}
    result = phrase_index.query(
        vector=_field(entry, 'values'),
        top_k=min(int(meta.get("total") or MAX_CACHED_PHRASES), MAX_CACHED_PHRASES),
        filter=query_filter,
        include_metadata=True,
        include_values=False,
    )
    texts = [(_field(m, 'metadata') or {}).get("text") for m in _field(result, 'matches') or []]
    return [t for t in texts if t]


def expand_word(word, model="gpt-4o-mini"):
    """
    Generate 4 close synonyms using OpenAI.
    Returns list with original word + 4 variations.
    """
    # Synonyms, not feelings: asking for "emotional words" turned e.g. "nerdy"
    # into "excited, passionate", so axes measured mood instead of meaning
    prompt = f"Give me 4 words or short phrases that mean nearly the same thing as '{word}' (close synonyms, not feelings it causes). Simply list them as a numbered list. Do not give me any other information."
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=100,
    )
    
    text = response.choices[0].message.content
    words = re.findall(r'^\d+\.\s*(.+)', text, re.MULTILINE)
    
    return [word] + [w.strip() for w in words if w.strip()]


def make_prompt_variants(word):
    """
    Generate diverse prompts for phrase generation.
    Expands word into 5 variants, creates 10 prompts per variant = 50 total prompts.
    Phrases are written as song lyrics in a fixed spread of styles and moods, so
    they look like the lyrics they're compared against and every word gets the
    same mix of tones (otherwise tone, not meaning, separates the two poles).
    """
    all_prompts = []
    
    words = expand_word(word, model="gpt-4o-mini")
    
    for w in words:
        all_prompts += [
            f"Write 5 rap lyric lines about being {w}. Format as a numbered list.",
            f"Write 5 pop song lyric lines where the singer is clearly {w}. Format as a numbered list.",
            f"Write 5 lines from a slow, sad song by someone who is {w}. Format as a numbered list.",
            f"Write 5 boastful, confident lyric lines from a singer who is {w}. Format as a numbered list.",
            f"Write 5 low-key, everyday lyric lines that quietly show someone is {w}, without using the word. Format as a numbered list.",
            f"Write 5 rock or punk lyric lines about a {w} life. Format as a numbered list.",
            f"Write 5 country or folk lyric lines describing someone {w}. Format as a numbered list.",
            f"Write 5 playful, funny lyric lines about being {w}. Format as a numbered list.",
            f"Write 5 lyric lines describing what a {w} person does on a typical night. Format as a numbered list.",
            f"Write 5 short song lyric lines that are unmistakably about being {w}. Format as a numbered list.",
        ]
    
    return all_prompts


def generate_dynamic_axis_phrases(word, model_name="gpt-4o-mini", max_workers=20):
    """
    Generate phrase embeddings for a given word using OpenAI.
    Checks Pinecone cache first, generates if not found.
    Returns numpy array of phrase embeddings.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Check Pinecone cache first
    cached = load_cached_phrases(word)
    if cached is not None:
        return cached
    
    print(f"Generating new phrases for '{word}'...")
    all_phrases = set()
    prompts = make_prompt_variants(word)
    
    def call_api(prompt):
        """Make single API call"""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=200,
            )
            
            text = response.choices[0].message.content
            phrases = re.findall(r'^\d+\.\s*(.+)', text, re.MULTILINE)
            phrases = [re.sub(r'[*"]', '', phrase) for phrase in phrases]
            phrases = [phrase.lower() for phrase in phrases]
            return [p.strip() for p in phrases if p.strip()]
        
        except Exception as e:
            print(f"Prompt failed: {e}")
            return []
    
    # Parallel execution for phrase generation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call_api, prompt): i for i, prompt in enumerate(prompts)}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            phrases = future.result()
            all_phrases.update(phrases)
            
            if completed % 10 == 0:
                print(f"  Progress: {completed}/{len(prompts)} prompts done, {len(all_phrases)} unique phrases so far")
    
    phrases_list = list(all_phrases)
    print(f"✓ Generated {len(phrases_list)} total phrases, now embedding them...")
    
    # Embed the phrases using OpenAI
    embeddings = get_openai_embeddings(phrases_list)
    
    # Save to Pinecone; the text is kept so people can see what a word means
    save_cached_phrases(word, phrases_list, embeddings)
    
    return embeddings