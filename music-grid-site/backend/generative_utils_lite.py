import re
import threading
import openai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import os
import json
from dotenv import load_dotenv

load_dotenv()


PHRASE_CACHE_DIR = 'cache/phrases'
os.makedirs(PHRASE_CACHE_DIR, exist_ok=True)

def load_cached_phrases(word):
    path = os.path.join(PHRASE_CACHE_DIR, f"{word.lower()}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def save_cached_phrases(word, phrases):
    path = os.path.join(PHRASE_CACHE_DIR, f"{word.lower()}.json")
    with open(path, "w") as f:
        json.dump(phrases, f, indent=2)


client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

@lru_cache(maxsize=100)
def expand_word(word, model):
    
    prompt = f"Give me 4 different emotional words closely related to '{word}'. Simply list the words. Do not give me any more information"


    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        top_p=0.95,
        max_tokens=600,
    )
    text = response.choices[0].message.content
    words = re.findall(r'^\d+\.\s*(.+)', text, re.MULTILINE)
    
    return [word] + words



def make_prompt_variants(word):
    words = expand_word(word, model="llama3-70b-8192")  # 5 related words
    prompts = []

    for w in words:
        prompt = f"""
        Write 25 emotionally expressive, vivid, and richly descriptive phrases that embody **extreme {w}**.

        Instructions:
        - Each phrase should be creative, metaphorical, and emotionally powerful
        - Avoid clichés (like "ray of sunshine", "top of the world")
        - Vary tone, structure, and perspective
        - Avoid using the word '{w}' in most phrases — describe the feeling without naming it directly
        - Format your response as a **numbered list**, 1 through 25

        Begin now:
        """
        prompts.append(prompt.strip())

    return prompts



def generate_dynamic_axis_phrases(word, model_name="llama3-70b-8192", max_workers = 4):

    cached = load_cached_phrases(word)
    if cached:
        return cached
    

    all_phrases = set()
    prompts = make_prompt_variants(word)

    def run_prompt(prompt):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                top_p=0.95,
                max_tokens=2000,
            )
            text = response.choices[0].message.content
            #phrases = re.findall(r"\d+\.\s*(?:\"|')?(.*?)(?:\"|')?\s*(?=\d+\.|$)", text, re.DOTALL)
            phrases = re.findall(r'^\d+\.\s*(.+)', text, re.MULTILINE)
            phrases = [re.sub(r'[*"]', '', phrase) for phrase in phrases]
            phrases = [phrase.lower() for phrase in phrases]
            return phrases

        except Exception as e:
            print(f"Prompt failed: {prompt}\nError: {e}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_prompt, prompt) for prompt in prompts]
        for future in as_completed(futures):
            all_phrases.update(future.result())

    save_cached_phrases(word, list(all_phrases))

    return list(all_phrases)


def generate_dynamic_axis_phrases_debug(word, model_name="llama3-70b-8192", max_workers = 4):
    all_phrases = set()
    prompts = make_prompt_variants(word)

    def run_prompt(prompt):
        try:
            thread_id = threading.get_ident()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                top_p=0.95,
                max_tokens=2000,
            )
            text = response.choices[0].message.content
            #phrases = re.findall(r"\d+\.\s*(?:\"|')?(.*?)(?:\"|')?\s*(?=\d+\.|$)", text, re.DOTALL)
            phrases = re.findall(r'^\d+\.\s*(.+)', text, re.MULTILINE)
            phrases = [re.sub(r'[*"]', '', phrase) for phrase in phrases]
            phrases = [phrase.lower() for phrase in phrases]

            print(f"[Thread {thread_id}] Completed prompt")
            return phrases

        except Exception as e:
            print(f"Prompt failed: {prompt}\nError: {e}")
            return []

    print(f"🔁 Submitting {len(prompts)} prompts using {max_workers} threads...")


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_prompt, prompt) for prompt in prompts]
        for future in as_completed(futures):
            all_phrases.update(future.result())
            
    print(f"✅ Finished generating phrases for '{word}' — total: {len(all_phrases)}")
    return list(all_phrases)