"""
Blocks slurs and hate speech in axis words, which other people can see in the
word list and phrase viewer. Song lyrics are deliberately not filtered.

Two layers:
  1. A fixed blocklist, matched after undoing common disguises (leetspeak,
     accents, stretched letters, spaces or punctuation between letters).
     Entries are stored as salted SHA-256 hashes so the repo doesn't spell the
     words out. To add one, run:
         python moderation_utils.py word <word>        (whole-word match)
         python moderation_utils.py substring <text>   (match anywhere)
     and paste the printed hash into the matching set below.
  2. OpenAI's moderation endpoint (free), for hate speech the list misses.
     Only hate and threat categories count: words like "horny" are fine here.
"""
import hashlib
import os
import re
import sys
import unicodedata
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LEET = str.maketrans({"0": "o", "1": "i", "!": "i", "|": "i", "3": "e", "4": "a",
                      "@": "a", "5": "s", "$": "s", "7": "t", "8": "b", "9": "g"})


def _hash(text):
    return hashlib.sha256(("compass-blocklist:" + text).encode()).hexdigest()[:16]


def _squash(text):
    """Squash every run of a repeated letter to one."""
    return re.sub(r"(.)\1+", r"\1", text)


# Matched anywhere in the squashed, space-free text, so entries are hashed squashed
BLOCKED_SUBSTRING_HASHES = {
    "13fed119597a8ca6", "2ae34f510c95c420", "2fdb538d5e880dfc", "4f1ff50a0fd56c5e",
    "51dd930db4ca85d6", "574a3b12a8164785", "67bfacfc45a28c5c", "6e49ac98520d4fba",
    "7ac495ee1ee51ec6", "7dcd1a949eeddf9e", "80b2ffcab3c93c7b", "86a453569649dffa",
    "919ac5eae9e7fe42", "99721f17fcbf4f8e", "9aa276ffa4737c6d", "a1641cd632f7aba2",
    "a22b49dd842df8dd", "ce6a79f2f28f9ffa", "e4067ba6a5f4dfb6", "e73e228fc4650bb7",
    "e80937428d0ad6ae", "ee761aa359bc8ef9", "ff75280e71e2a249",
}
SUBSTRING_LENGTHS = {4, 5, 6, 7, 8, 9, 10, 11, 12}  # lengths of the entries above

# Real words that contain a blocked substring once squashed. Checked before
# squashing, so a slur that squashes to the same letters isn't let through.
ALLOWED_WORDS = {"niger", "nigeria", "nigerian", "nigerians", "snigger", "sniggers", "sniggering"}

# Short or ambiguous terms, matched only as whole words so ordinary words containing
# them are fine. Compared as typed and with runs of 3+ letters cut to 2.
BLOCKED_WORD_HASHES = {
    "078ae8e9fcbcb210", "0a1889d55cc39a81", "2dd62c653b69508e", "2f7e27a6b559487b",
    "4198775db3194ee2", "42e5098dd7cfaa07", "474b351c613893e5", "4d5cd5dcd65221e0",
    "4f2e2f16c34555be", "5344ea88ff1aa4d7", "579539a02f54126e", "59e63b689cb1c40b",
    "5c1d4438f18701ef", "67352e9144514243", "74923219a685a69a", "761817f011552d1b",
    "7737ad03c2f80180", "79b5b558b187b996", "976df43325c28a38", "9baab1113f04669e",
    "9f54428a13511001", "9f987d71566f4d8e", "a5dbc414b08bd5ad", "a97243ca53f50e7e",
    "bd8f8a5d82f9eb06", "c31d461835338726", "d74b3d7a40aaf455", "daa7903e6513c1a9",
    "dc2141adcf4cb094", "e4d8c6bef5198744", "eac507f9b0476d57", "ec494630960fcc83",
    "ecd3913d5421e678", "f27c3c2a10be1098", "fb9369aa447cde55",
}

# Whole words with no doubled letters, which can also be matched fully squashed
SQUASHABLE_WORD_HASHES = {
    "078ae8e9fcbcb210", "0a1889d55cc39a81", "2dd62c653b69508e", "2f7e27a6b559487b",
    "4198775db3194ee2", "42e5098dd7cfaa07", "474b351c613893e5", "4d5cd5dcd65221e0",
    "5344ea88ff1aa4d7", "579539a02f54126e", "59e63b689cb1c40b", "67352e9144514243",
    "74923219a685a69a", "761817f011552d1b", "79b5b558b187b996", "976df43325c28a38",
    "9baab1113f04669e", "9f54428a13511001", "9f987d71566f4d8e", "a97243ca53f50e7e",
    "bd8f8a5d82f9eb06", "c31d461835338726", "d74b3d7a40aaf455", "daa7903e6513c1a9",
    "e4d8c6bef5198744", "eac507f9b0476d57", "ec494630960fcc83", "ecd3913d5421e678",
    "f27c3c2a10be1098",
}

MODERATION_CATEGORIES = ["hate", "hate/threatening", "harassment/threatening"]


def _letters(text):
    """Lowercase, strip accents, undo leetspeak, keep only letters and spaces."""
    text = unicodedata.normalize("NFKD", text.lower().translate(LEET))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]|[\d_]", "", text)


def _on_blocklist(word):
    raw = _letters(word).split()
    if any(_hash(t) in BLOCKED_WORD_HASHES for t in raw):
        return True
    tokens = [re.sub(r"(.)\1{2,}", r"\1\1", t) for t in raw]
    # A word spelled out one letter at a time
    if len(tokens) > 1 and all(len(t) == 1 for t in tokens):
        tokens = ["".join(tokens)]
    if any(_hash(t) in BLOCKED_WORD_HASHES or _hash(_squash(t)) in SQUASHABLE_WORD_HASHES for t in tokens):
        return True

    # Joined before squashing so multi-word forms are caught as well
    joined = _squash("".join(t for t in tokens if t not in ALLOWED_WORDS))
    return any(
        _hash(joined[i:i + n]) in BLOCKED_SUBSTRING_HASHES
        for n in SUBSTRING_LENGTHS
        for i in range(len(joined) - n + 1)
    )


def _flagged_by_moderation(word):
    try:
        result = client.moderations.create(model="omni-moderation-latest", input=word).results[0]
    except Exception as e:
        # Fail open: the blocklist has already run, and a moderation outage
        # shouldn't stop people building compasses
        print(f"Moderation check failed for '{word}': {e}", flush=True)
        return False
    categories = result.categories.model_dump(by_alias=True)
    return any(categories.get(c) for c in MODERATION_CATEGORIES)


def is_blocked_word(word, use_moderation=True):
    """True if an axis word is a slur or hate speech."""
    if _on_blocklist(word):
        return True
    return use_moderation and _flagged_by_moderation(word)


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] in ("word", "substring"):
    entry = _letters(sys.argv[2]).replace(" ", "")
    if sys.argv[1] == "substring":
        entry = _squash(entry)
        print(f"Add to BLOCKED_SUBSTRING_HASHES: {_hash(entry)}  (and {len(entry)} to SUBSTRING_LENGTHS)")
    else:
        print(f"Add to BLOCKED_WORD_HASHES: {_hash(entry)}")
        if _squash(entry) == entry:
            print(f"Also add to SQUASHABLE_WORD_HASHES: {_hash(entry)}")
