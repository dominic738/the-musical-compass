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
ALLOWED_WORDS = {"niger", "nigeria", "nigerian", "nigerians", "snigger", "sniggers", "sniggering",
                 "scunthorpe", "swank", "swanky"}

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

# Crude but not hateful: allowed on a compass, kept out of the public word list.
# Same hashing and matching as the blocklist above.
CRUDE_SUBSTRING_HASHES = {
    "07ce604cb27afed2", "09c1ee14564b9589", "2ce37f9399ed3a70", "53270cbac4785080",
    "7ca445110d23b977", "d0d1abd3a4861b93", "fba35942a05edcb0",
}
CRUDE_SUBSTRING_LENGTHS = {3, 4}
CRUDE_WORD_HASHES = {
    "08ae9d7861402fc0", "0f563ab17f410697", "115f01d686599b3f", "1390fe86996f77fa",
    "193865cc2cb45cf8", "233490f270cac7bd", "3808fa2eb4071eae", "4199f248003c9fa8",
    "476c0ddb7342b162", "4ef4fffdbb8e016b", "549ad48b4d589c7c", "56cc5d7817f3a4a0",
    "5cfa400d7ea3889b", "6701269369cad66d", "7a83cc21de59aa31", "7c3efaef15b1cbfd",
    "7ca8d6002e07d04a", "7ec84024385a14b2", "81efe7ea031d6de6", "8286a1ae3ccfffab",
    "86d60601392d3278", "8f7457d2a48f0117", "a441cd3a8ab85577", "a8520c0eb126da55",
    "aaca8c5a79d402e0", "b9a53c5d129c2d4e", "c1de25db06ce3ecc", "d20e05dda6f3769c",
    "d4a4f9f148f2cfcf", "d5cd12713212f2cf", "d837e469709bea34", "dd35cd11b018ef99",
    "f0cf30e6c02281d8",
}
CRUDE_SQUASHABLE_HASHES = {
    "08ae9d7861402fc0", "0f563ab17f410697", "115f01d686599b3f", "1390fe86996f77fa",
    "233490f270cac7bd", "3808fa2eb4071eae", "4199f248003c9fa8", "4ef4fffdbb8e016b",
    "56cc5d7817f3a4a0", "5cfa400d7ea3889b", "6701269369cad66d", "7ca8d6002e07d04a",
    "7ec84024385a14b2", "81efe7ea031d6de6", "86d60601392d3278", "8f7457d2a48f0117",
    "a8520c0eb126da55", "aaca8c5a79d402e0", "d20e05dda6f3769c", "d4a4f9f148f2cfcf",
    "d5cd12713212f2cf", "d837e469709bea34", "dd35cd11b018ef99", "f0cf30e6c02281d8",
}

# The public word list also hides words moderation scores as sexual or insulting.
# Tuned on real words: "horny" scores 0.57 sexual and "lesbian" 0.21, so 0.25
# hides the first without hiding identity words.
PUBLIC_SEXUAL_MAX = 0.25
PUBLIC_HARASSMENT_MAX = 0.15


def _letters(text):
    """Lowercase, strip accents, undo leetspeak, keep only letters and spaces."""
    text = unicodedata.normalize("NFKD", text.lower().translate(LEET))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]|[\d_]", "", text)


def _matches(word, word_hashes, squashable_hashes, substring_hashes, substring_lengths):
    raw = _letters(word).split()
    if any(_hash(t) in word_hashes for t in raw):
        return True
    tokens = [re.sub(r"(.)\1{2,}", r"\1\1", t) for t in raw]
    # A word spelled out one letter at a time
    if len(tokens) > 1 and all(len(t) == 1 for t in tokens):
        tokens = ["".join(tokens)]
    if any(_hash(t) in word_hashes or _hash(_squash(t)) in squashable_hashes for t in tokens):
        return True

    # Joined before squashing so multi-word forms are caught as well
    joined = _squash("".join(t for t in tokens if t not in ALLOWED_WORDS))
    return any(
        _hash(joined[i:i + n]) in substring_hashes
        for n in substring_lengths
        for i in range(len(joined) - n + 1)
    )


def _on_blocklist(word):
    return _matches(word, BLOCKED_WORD_HASHES, SQUASHABLE_WORD_HASHES,
                    BLOCKED_SUBSTRING_HASHES, SUBSTRING_LENGTHS)


def _is_crude(word):
    return _matches(word, CRUDE_WORD_HASHES, CRUDE_SQUASHABLE_HASHES,
                    CRUDE_SUBSTRING_HASHES, CRUDE_SUBSTRING_LENGTHS)


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


_public_cache = {}


def is_public_word(word):
    """
    Whether a word is clean enough to show strangers in the word list. Words
    that fail can still be used; they just aren't advertised. Fails closed:
    if moderation can't be reached, the word stays hidden until next time.
    """
    key = word.lower()
    if key in _public_cache:
        return _public_cache[key]
    if _on_blocklist(key) or _is_crude(key):
        _public_cache[key] = False
        return False
    try:
        result = client.moderations.create(model="omni-moderation-latest", input=key).results[0]
    except Exception as e:
        print(f"Moderation check failed for '{key}': {e}", flush=True)
        return False
    scores = result.category_scores.model_dump(by_alias=True)
    public = not (result.flagged
                  or scores.get("sexual", 0) >= PUBLIC_SEXUAL_MAX
                  or scores.get("harassment", 0) >= PUBLIC_HARASSMENT_MAX)
    _public_cache[key] = public
    return public


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] in ("word", "substring"):
    entry = _letters(sys.argv[2]).replace(" ", "")
    if sys.argv[1] == "substring":
        entry = _squash(entry)
        print(f"Add to BLOCKED_SUBSTRING_HASHES: {_hash(entry)}  (and {len(entry)} to SUBSTRING_LENGTHS)")
    else:
        print(f"Add to BLOCKED_WORD_HASHES: {_hash(entry)}")
        if _squash(entry) == entry:
            print(f"Also add to SQUASHABLE_WORD_HASHES: {_hash(entry)}")
