import re
import openai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.get_env("OPENAI_KEY")

client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


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

    all_prompts = []
    
    words = expand_word(word, model="llama3-70b-8192")

    for word in words:

    
        all_prompts += [
            f"Give me 5 short, emotionally raw phrases expressing extreme {word}. Format as a numbered list.",
            f"Write 5 expressive slang phrases that capture extreme {word}, like something said on social media.",
            f"Write 5 poetic phrases that convey intense {word}. Avoid clichés.",
            f"Write 5 brutally honest things someone might say to a friend when feeling extreme {word}.",
            f"Write 5 dramatic declarations of {word}, like someone screaming it in a movie.",
            f"Give 5 diary-style phrases capturing the feeling of overwhelming {word}.",
            f"Write 5 lines that describe what extreme {word} feels like, using metaphors or body sensations.",
            f"What would someone mutter to themselves under their breath if they were overcome with {word}? Give 5 phrases.",
            f"Write 5 short lines that someone might text to a friend in the middle of feeling extreme {word}.",
            f"Give 5 phrases that capture intense {word} without using the actual word '{word}'.",
        ]

    return all_prompts




def generate_dynamic_axis_phrases(word, model_name="llama3-70b-8192"):
    all_phrases = set()
    prompts = make_prompt_variants(word)

    for prompt in prompts:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                top_p=0.95,
                max_tokens=600,
            )
            text = response.choices[0].message.content
            #phrases = re.findall(r"\d+\.\s*(?:\"|')?(.*?)(?:\"|')?\s*(?=\d+\.|$)", text, re.DOTALL)
            phrases = re.findall(r'^\d+\.\s*(.+)', text, re.MULTILINE)
            phrases = [re.sub(r'[*"]', '', phrase) for phrase in phrases]
            phrases = [phrase.lower() for phrase in phrases]
            all_phrases.update(p.strip() for p in phrases if p.strip())

        except Exception as e:
            print(f"Prompt failed: {prompt}\nError: {e}")

    return list(all_phrases)