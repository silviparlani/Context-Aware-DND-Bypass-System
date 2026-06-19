"""Scrub PII token strings out of a fitted TfidfVectorizer — in place.

The vectorizer's ``vocabulary_`` dict stores every learned token as plaintext,
including real names and phone numbers harvested from the raw chat bodies. This
script replaces each PII token's *key string* with an inert placeholder while
preserving its integer column index, so:

  * the vocabulary size (and therefore the RF model's expected feature count)
    is unchanged -> NO retrain, NO refit required;
  * the IDF weights are untouched;
  * the PII strings are genuinely removed from the pickle (not encrypted/hidden).

The only behavioural change is that a redacted token will no longer activate its
column for future messages that happen to contain it. For a DND classifier those
tokens are noise, so predictions on real traffic are effectively unchanged.

Run:  python src/scrub_vectorizer.py
The list of name tokens is read from data/pii_tokens.txt (gitignored) so the
names themselves never land in version control. Phone numbers (>=7 consecutive
digits) are detected automatically.
"""

import os
import re
import shutil

import joblib

ROOT = os.path.dirname(os.path.dirname(__file__))
VECTORIZER_PATH = os.path.join(ROOT, "models", "dnd_vectorizer.pkl")
PII_LIST_PATH = os.path.join(ROOT, "data", "pii_tokens.txt")
BACKUP_PATH = os.path.join(ROOT, "data", "dnd_vectorizer.preredact.bak.pkl")

PHONE_RE = re.compile(r"\d{7,}")


def load_name_tokens(path):
    if not os.path.exists(path):
        return set()
    tokens = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                tokens.add(line.lower())
    return tokens


def main():
    vec = joblib.load(VECTORIZER_PATH)
    vocab = vec.vocabulary_
    original_size = len(vocab)

    names = load_name_tokens(PII_LIST_PATH)
    phones = {t for t in vocab if PHONE_RE.search(t)}
    to_redact = (names & set(vocab)) | phones

    if not to_redact:
        print("No PII tokens found in vocabulary. Nothing to do.")
        return

    # Back up the original (with PII) to a gitignored location before mutating.
    shutil.copy2(VECTORIZER_PATH, BACKUP_PATH)
    print(f"Backed up original (contains PII) -> {BACKUP_PATH}")

    print(f"\nRedacting {len(to_redact)} tokens "
          f"({len(phones)} phone-like, {len(to_redact) - len(phones)} named):")
    for tok in sorted(to_redact):
        label = "phone" if tok in phones else "name "
        idx = vocab[tok]
        new_key = f"__redacted_{idx}__"
        del vocab[tok]
        vocab[new_key] = idx
        print(f"  [{label}] {tok!r:>16} -> {new_key}  (col {idx})")

    # Integrity checks: same size, same set of column indices, no collisions.
    assert len(vocab) == original_size, "vocabulary size changed!"
    assert len(set(vocab.values())) == original_size, "column index collision!"
    assert not any(PHONE_RE.search(t) for t in vocab), "phone token survived!"
    assert not (set(vocab) & names), "named token survived!"

    joblib.dump(vec, VECTORIZER_PATH)
    print(f"\nSaved scrubbed vectorizer -> {VECTORIZER_PATH}")
    print(f"Vocabulary size unchanged: {original_size}")

    # Prove transform still produces the same-width feature vector.
    width = vec.transform(["quick sanity check message"]).shape[1]
    print(f"transform() output width: {width} (model expects this + 4 metadata)")


if __name__ == "__main__":
    main()
