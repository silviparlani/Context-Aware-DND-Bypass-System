# Vectorization

---

## 1. What Is Vectorization and Why Do We Need It Here

Machine learning models operate on numbers, not strings. Vectorization is the process of converting raw text into a numerical representation — a vector of features — that a model can learn from.

In this project, the `message` column is the primary text signal for classifying whether a message is urgent (label `1`) or non-urgent (label `0`). Before any classifier can be trained, every message must be mapped to a fixed-length numerical vector. Without this step, the model has nothing to compute on.

Why it matters specifically for DND bypass:

- Messages are short and informal (WhatsApp chats), so the exact words used carry most of the signal.
- Urgent messages tend to use specific vocabulary: *"call me"*, *"SOS"*, *"ASAP"*, *"help"*, *"emergency"*. A good vectorizer should surface these words as discriminative.
- The dataset is small (500-row sample for development), so the vectorization method needs to work well in low-data regimes without requiring a large pretrained corpus.

---

## 2. Vectorizers — Comparison

### Bag of Words (CountVectorizer)

Counts how many times each word appears in a message. Each unique word in the vocabulary becomes a feature column.

| | |
|---|---|
| **Pros** | Simple; fast; interpretable; no external data needed |
| **Cons** | Treats all words as equally important; common words (e.g., *"the"*, *"I"*, *"and"*) dominate; no notion of word rarity across the corpus |

### TF-IDF (TfidfVectorizer)

Extends Bag of Words by weighting each word count by how rare that word is across all documents. Words that appear in almost every message get downweighted; words unique to a few messages get upweighted.

| | |
|---|---|
| **Pros** | Suppresses common noise words automatically; upweights rare discriminative words; interpretable; fast; no external data needed |
| **Cons** | Still ignores word order and sentence structure; treats synonyms as unrelated features (e.g., *"urgent"* and *"ASAP"* are independent features) |

### Word2Vec / GloVe (Static Word Embeddings)

Pretrained dense vector representations trained on large corpora. Each word maps to a fixed vector capturing semantic similarity (e.g., *"king"* − *"man"* + *"woman"* ≈ *"queen"*).

| | |
|---|---|
| **Pros** | Captures semantic relationships; dense (lower-dimensional); handles synonyms better |
| **Cons** | Requires averaging word vectors to get a document vector, losing structure; pretrained on formal text — may not align well with informal WhatsApp language; no frequency signal; OOV (out-of-vocabulary) words like *"heyyyy"* are dropped |

### FastText

Similar to Word2Vec but represents words as bags of character n-grams, allowing it to handle typos and informal spellings by decomposing unknown words into subword units.

| | |
|---|---|
| **Pros** | Handles OOV words and typos well — relevant for WhatsApp messages; can be trained on your own corpus |
| **Cons** | More complex to train from scratch; pretrained models still carry the formal-text bias; still loses frequency signal |

### BERT / Sentence Transformers (Contextual Embeddings)

Transformer-based models that produce a contextual embedding for each message, meaning the same word gets a different vector depending on surrounding words.

| | |
|---|---|
| **Pros** | State-of-the-art semantic understanding; handles context, negation, and polysemy |
| **Cons** | Heavy — slow to run without a GPU; significant overkill for a baseline on 500 rows; harder to interpret; adds model complexity before simpler baselines have been validated |

### HashingVectorizer

Maps words to feature indices using a hash function instead of a learned vocabulary.

| | |
|---|---|
| **Pros** | Memory efficient; no vocabulary to store; fast on streaming data |
| **Cons** | Hash collisions reduce quality; not interpretable at all; no IDF weighting available in its basic form |

---

## 3. Why TF-IDF Is the Right Choice for the Baseline

**The core requirement of a baseline is to be simple, fast, and interpretable — and TF-IDF ticks all three.**

### It naturally aligns with how urgency works in text

Urgent messages in this dataset tend to use specific, rare vocabulary. Words like *"SOS"*, *"emergency"*, *"help"*, *"ASAP"*, *"please call"* don't appear in every message — which is exactly the condition under which TF-IDF assigns high weight. Meanwhile, words like *"okay"*, *"haha"*, *"I"*, *"the"* appear everywhere and get suppressed. This behaviour is precisely what we want: urgency keyword detection with noise reduction baked in.

### The synthetic labels reinforce this

The urgent messages (label `1`) were generated from 55 emergency templates with specific phrasing. TF-IDF will assign high scores to those exact terms because they are rare relative to the 93K-row corpus they were injected into. This makes the feature–label relationship particularly clean for an initial model.

### It works well in low-data regimes

With only 500 rows (36 positives), there isn't enough data to fine-tune embeddings or benefit from the contextual nuance of transformers. TF-IDF produces meaningful features immediately with no training data overhead beyond the vectorizer's fit step.

### It pairs well with strong baseline classifiers

TF-IDF sparse vectors work directly with Logistic Regression, Multinomial Naive Bayes, and Linear SVM — all of which are standard, fast, and highly interpretable baseline choices that perform well on text classification tasks of this scale.

### It is interpretable

After training, you can inspect which words have the highest TF-IDF weights for each class. This directly validates (or challenges) your intuitions about what makes a message urgent — something dense embeddings cannot easily offer at this stage.

### When to graduate beyond TF-IDF

Once the baseline is validated and the signal structure is better understood, the logical next step is to try FastText (to handle informal spellings) or a lightweight sentence transformer (to capture semantic relationships like *"not okay"* vs *"okay"*). But those are experiments for after TF-IDF has set the performance floor.
