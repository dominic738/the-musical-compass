"""
Review axis words before they appear in the public word list. Run locally
(it needs the Pinecone and OpenAI keys in .env):

    python review_words.py                      words waiting for review, with what the filters think
    python review_words.py all                  every word and its status
    python review_words.py approve <word> ...   show in the public list
    python review_words.py hide <word> ...      keep it usable, but never show it publicly
    python review_words.py reject <word> ...    hide, and refuse it as an axis word from now on
    python review_words.py pending <word> ...   undo: back to waiting

Quote words with spaces: python review_words.py approve "main character"
"""
import sys
import generative_utils as cu
import moderation_utils as mu

COMMANDS = {"approve": cu.REVIEW_APPROVED, "hide": cu.REVIEW_HIDDEN,
            "reject": cu.REVIEW_REJECTED, "pending": cu.REVIEW_PENDING}


def filter_notes(word):
    notes = []
    if mu._on_blocklist(word):
        notes.append("BLOCKLIST")
    if mu.flagged_as_slur(word):
        notes.append("AI: SLUR")
    if mu._flagged_by_moderation(word):
        notes.append("moderation: hate")
    if mu._is_crude(word) or not mu.is_public_word(word):
        notes.append("crude (hidden even if approved)")
    return ", ".join(notes) or "no flags"


def show_pending(reviews):
    pending = sorted(w for w, s in reviews.items() if s == cu.REVIEW_PENDING)
    if not pending:
        print("Nothing waiting for review.")
        return
    print(f"{len(pending)} waiting for review:\n")
    for word in pending:
        print(f"  {word:<24} {filter_notes(word)}")
    print('\nApprove with: python review_words.py approve <word> ...')


def main(args):
    reviews = cu.all_word_reviews()
    if not args:
        show_pending(reviews)
    elif args == ["all"]:
        for status in (cu.REVIEW_APPROVED, cu.REVIEW_HIDDEN, cu.REVIEW_PENDING, cu.REVIEW_REJECTED):
            words = sorted(w for w, s in reviews.items() if s == status)
            print(f"{status} ({len(words)}): {', '.join(words) or '-'}")
    elif args[0] in COMMANDS and len(args) > 1:
        for word in (w.strip().lower() for w in args[1:]):
            if word not in reviews:
                print(f"  {word}: no such word, skipped")
                continue
            cu.set_review(word, COMMANDS[args[0]])
            print(f"  {word}: {COMMANDS[args[0]]}")
        print("The live list updates within a minute.")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
