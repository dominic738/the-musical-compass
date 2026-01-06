import re
import openai
import os
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


def load_cached_phrases(word):
    """
    Load cached phrase embeddings from Pinecone.
    Returns numpy array of embeddings if found, None otherwise.
    """
    try:
        # Try to fetch vectors by ID prefix (happy_0, happy_1, etc.)
        # Pinecone doesn't support prefix search, so we try common range
        ids_to_fetch = [f"{word.lower()}_{i}" for i in range(500)]  # Max 500 phrases per word
        
        result = phrase_index.fetch(ids=ids_to_fetch)
        
        if result and result.get('vectors'):
            embeddings = [vec['values'] for vec in result['vectors'].values()]
            if embeddings:
                embeddings_array = np.array(embeddings)
                print(f"✓ Loaded {len(embeddings)} cached phrase embeddings for '{word}' from Pinecone")
                return embeddings_array
    except Exception as e:
        print(f"Pinecone fetch error for '{word}': {e}")
    return None


def save_cached_phrases(word, embeddings):
    """
    Save phrase embeddings to Pinecone.
    Each embedding gets stored as a separate vector.
    """
    try:
        vectors = []
        for i, embedding in enumerate(embeddings):
            vectors.append({
                "id": f"{word.lower()}_{i}",
                "values": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                "metadata": {
                    "word": word.lower()
                }
            })
        
        # Upsert in batches of 100 (Pinecone limit)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            phrase_index.upsert(vectors=batch)
        
        print(f"✓ Cached {len(embeddings)} phrase embeddings for '{word}' in Pinecone")
    except Exception as e:
        print(f"Pinecone upsert error for '{word}': {e}")


def expand_word(word, model="gpt-4o-mini"):
    """
    Generate 4 related emotional words using OpenAI.
    Returns list with original word + 4 variations.
    """
    prompt = f"Give me 4 different emotional words closely related to '{word}'. Simply list the words as a numbered list. Do not give me any other information."
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=100,
    )
    
    text = response.choices[0].message.content
    words = re.findall(r'^\d+\.\s*(.+)', text, re.MULTILINE)
    
    return [word] + words


def make_prompt_variants(word):
    """
    Generate diverse prompts for phrase generation.
    Expands word into 5 variants, creates 10 prompts per variant = 50 total prompts.
    """
    all_prompts = []
    
    words = expand_word(word, model="gpt-4o-mini")
    
    for w in words:
        all_prompts += [
            f"Give me 5 short, emotionally raw phrases expressing extreme {w}. Format as a numbered list.",
            f"Write 5 expressive slang phrases that capture extreme {w}, like something said on social media.",
            f"Write 5 poetic phrases that convey intense {w}. Avoid clichés.",
            f"Write 5 brutally honest things someone might say to a friend when feeling extreme {w}.",
            f"Write 5 dramatic declarations of {w}, like someone screaming it in a movie.",
            f"Give 5 diary-style phrases capturing the feeling of overwhelming {w}.",
            f"Write 5 lines that describe what extreme {w} feels like, using metaphors or body sensations.",
            f"What would someone mutter to themselves under their breath if they were overcome with {w}? Give 5 phrases.",
            f"Write 5 short lines that someone might text to a friend in the middle of feeling extreme {w}.",
            f"Give 5 phrases that capture intense {w} without using the actual word '{w}'.",
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
    
    # Save embeddings to Pinecone (don't need to save phrases)
    save_cached_phrases(word, embeddings)
    
    return embeddings