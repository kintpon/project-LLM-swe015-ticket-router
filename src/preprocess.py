"""Area B - classical NLP: regex cleaning, model-code extraction, Levenshtein spelling correction."""
import re
from functools import lru_cache

CODE_RE = re.compile(r"\b[A-Za-z]{2,3}-?\d{3,5}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
REPEAT_RE = re.compile(r"([a-z])\1{2,}")  # letters only, so numbers like 1000 stay intact

# product words we are willing to correct towards
LEXICON = ["dishwasher", "washing", "machine", "oven", "fridge", "refrigerator", "dryer",
           "freezer", "microwave"]


def levenshtein(a, b):
    """Edit distance by dynamic programming (insert, delete, substitute = 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def correct_token(token, vocab):
    """Return a lexicon word if the token looks like a misspelling of it, else the token."""
    if len(token) < 5 or token in vocab or token in LEXICON:
        return token  # guard: short or known words ("over", "even") are never touched
    best = min(LEXICON, key=lambda w: levenshtein(token, w))
    return best if levenshtein(token, best) <= 2 else token


@lru_cache(maxsize=1)
def get_vocab():
    """Training vocabulary: every word in the CLINC train rows. The appliance rows are left out
    because they contain deliberate typos ("dishwaser") that must stay correctable."""
    from src.data import load_split
    df = load_split("train")
    texts = df[df["source"] == "clinc"]["text"]
    return frozenset(t for x in texts for t in re.findall(r"[a-z']+", x.lower()))


def preprocess(text, vocab=None):
    """Clean a message. Returns (clean_text, model_codes, corrections)."""
    if vocab is None:
        vocab = get_vocab()
    codes = CODE_RE.findall(text)
    text = CODE_RE.sub(" ", text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = REPEAT_RE.sub(r"\1\1", text)  # soooo -> soo
    tokens, corrections = [], []
    for tok in re.findall(r"[a-z']+|\d+", text):
        fixed = correct_token(tok, vocab)
        if fixed != tok:
            corrections.append((tok, fixed))
        tokens.append(fixed)
    return " ".join(tokens), codes, corrections
