# Baseline Models

---

## 1. The Model Landscape and Where This Project Sits

Before picking a model, it helps to understand the categories they fall into — because the right choice depends heavily on what kind of data you have, how much of it you have, and what question you're trying to answer.

### The Broad Categories

**Probabilistic models** estimate the likelihood of a class directly from word frequencies. They are the simplest NLP classifiers and require almost no setup. Naive Bayes is the canonical example.

**Linear models** learn a weighted sum of features and draw a decision boundary between classes. They are fast, interpretable, and handle high-dimensional sparse data exceptionally well. Logistic Regression is the standard.

**Ensemble / tree-based models** build collections of decision trees that each ask binary questions about features and vote together. They capture non-linear relationships and interactions between features. Random Forest and XGBoost fall here.

**Neural networks** learn hierarchical representations of data through layers of transformations. They don't need hand-engineered features but are data-hungry and slow to iterate on.

**Transformers** are a specific family of neural networks (BERT, RoBERTa, etc.) that model language using attention mechanisms, understanding words in context rather than in isolation. State-of-the-art for NLP, but computationally expensive.

### What Category This Project Selects From — and Why

The baseline lives in the **linear model** category, with probabilistic models as a comparison point.

The reasons are grounded in the project's current state:

- The working dataset is 500 rows with 36 positive examples (label `1`). That rules out neural networks and transformers — they need far more data to learn meaningful representations.
- The features are TF-IDF vectors: large, sparse, and high-dimensional. Linear models were essentially designed for this input format.
- The goal at baseline is to answer one question: *do the words themselves signal urgency?* A linear model answers this cleanly and interpretably.
- Tree-based models (Random Forest, XGBoost) become relevant in a later phase when metadata features — sender identity, time of day, message frequency — are added alongside text. They shine at capturing interactions between mixed feature types. That's a Week 4 concern.

---

## 2. How Each Model Works

### Naive Bayes

Naive Bayes is a probabilistic classifier grounded in Bayes' theorem. During training it estimates two things from the data: how often each class appears (prior probability), and how often each word appears given each class (likelihood). At prediction time it multiplies these together to compute the probability of each class given the words in the message, then picks the class with the higher probability.

The "naive" part refers to its core assumption: every word is treated as statistically independent of every other word. The probability of a message containing *"call me now"* is computed as:

```
P("call") × P("me") × P("now") | class
```

This independence assumption is obviously false — real language has deep dependencies between words — but it turns out not to matter much in practice for short text classification. Because it only needs to count word frequencies, it trains in a single pass over the data and generalises surprisingly well on small datasets.

Its main weakness is that probability estimates are poorly calibrated. It's good at ranking which class is more likely but bad at telling you *how* likely. It also treats very common words as informative unless stopwords are removed explicitly, though pairing it with TF-IDF mitigates this.

**In this project:** useful as a sanity-check baseline and lower-bound comparison. If Logistic Regression doesn't clearly beat it, the feature engineering needs revisiting.

---

### Logistic Regression

Despite the name, Logistic Regression is a classification algorithm. It learns a weight for every feature in the input vector. Each weight represents how strongly that feature pushes a message toward urgent (positive weight) or non-urgent (negative weight). During prediction it computes a weighted sum of all features, passes it through a sigmoid function to squash the result to a probability between 0 and 1, and classifies the message based on a threshold (typically 0.5).

For a TF-IDF input the model is learning something like:

| Feature | Learned Weight |
|---|---|
| `urgent` | +2.1 |
| `asap` | +1.8 |
| `help` | +1.4 |
| `meeting` | −0.7 |
| `haha` | −1.5 |

The final prediction is: does the weighted combination of all words in this message push past the decision boundary?

Training happens via gradient descent — the model iteratively adjusts weights to minimise a loss function (log loss / cross-entropy) on the training set. Regularisation (L1 or L2) is applied to prevent the model from overfitting to rare words.

**In this project:** the primary baseline model. Covered in detail in section 3.

---

### Random Forest

A Random Forest is an ensemble of decision trees. Each individual tree is trained on a random subset of the data and at each split considers only a random subset of features. This randomness decorrelates the trees so that their errors don't compound — when you average their votes, noise cancels out and signal accumulates.

Each decision tree works by recursively splitting data on the feature that best separates the classes. In the context of this project, an individual tree might learn:

```
IF sender = Person_7
  AND hour < 5
  AND message contains "call"
THEN label = 1
```

That's a three-way interaction. Logistic Regression can't learn that without explicit feature engineering; a tree discovers it automatically.

Random Forests handle mixed data types (text features alongside numeric metadata), non-linear relationships, and are robust to outliers and irrelevant features. Their main weakness for pure TF-IDF text is that the high dimensionality of the vocabulary space dilutes the random feature sampling — most sampled features at any split will be zero for a given message. They work much better once lower-dimensional metadata features are in the mix.

**In this project:** the right tool once sender identity, time-of-day, and message frequency features are added alongside TF-IDF. Not the baseline; a Week 4 upgrade.

---

### XGBoost / Gradient Boosting

Where Random Forest builds trees in parallel and averages them, Gradient Boosting builds trees sequentially. Each new tree specifically targets the residual errors of all previous trees — it learns to fix what the current ensemble is getting wrong. The result is a highly accurate model that squeezes more signal out of structured/tabular data than almost anything else.

XGBoost is the industry-standard implementation, widely used in fraud detection, search ranking, and recommendation systems because it delivers excellent performance on engineered tabular features with relatively fast training.

Its weaknesses are that it requires more careful hyperparameter tuning and is more prone to overfitting than Random Forest without regularisation. It's also harder to debug when something goes wrong.

**In this project:** a potential extension after Random Forest, if more accuracy is needed and the feature set is mature. Not needed for the current scope.

---

### Neural Networks

Neural networks learn representations of data through layers of parameterised transformations. Rather than hand-engineering features, they learn which combinations of raw input patterns are predictive. Deeper networks can model increasingly abstract relationships.

For text, a simple neural network would take word embeddings as input and learn to combine them into a document representation. More sophisticated architectures (LSTMs, CNNs over text) can capture sequence structure and local patterns.

The core constraint is data. Neural networks have many parameters to learn and need enough examples to avoid overfitting to noise. With 500 rows and 36 positives, there isn't close to enough data. A neural network trained here would likely memorise the training set and generalise poorly.

**In this project:** not relevant at current scale.

---

### Transformers (BERT and variants)

Transformers model language using an attention mechanism that allows every word in a sentence to attend to every other word simultaneously. This means word representations are contextual — the same word gets a different vector depending on the surrounding sentence.

The practical consequence is that BERT can distinguish:

> "call me when you're free"

from:

> "call emergency services"

where a TF-IDF model sees both as containing the feature `call` and weights them identically.

Pre-trained transformer models (BERT, RoBERTa, DistilBERT) have seen hundreds of billions of words and encode broad language understanding. Fine-tuning on a downstream task like urgency classification then adjusts these weights for the specific problem.

The cost is computational. Even inference (not training) is significantly slower than Logistic Regression, and fine-tuning requires a GPU and more data than this project currently has to realise its potential.

**In this project:** mentioned in the future work section. A plausible extension if the dataset is scaled up or if the production system needs higher precision on borderline cases.

---

## 3. Why Logistic Regression Is the Right Baseline

### It matches the input format exactly

TF-IDF vectors are large, sparse, and high-dimensional — often tens of thousands of features, most of which are zero for any given message. Logistic Regression was designed for exactly this regime. It fits efficiently because it only needs to update weights for non-zero features in each training example, and sparse matrix operations are fast. There's no dimensionality mismatch, no preprocessing gap, no wasted compute.

### It directly answers the baseline question

The baseline exists to answer one specific question: *do the words in a message signal urgency?* Logistic Regression answers this directly — it learns a weight per word, and prediction is a weighted vote of all words present. There are no interaction terms, no structural assumptions about the data, no latent representations. The signal path from feature to prediction is transparent.

### The weights are immediately interpretable

After training, inspecting the highest positive-weight features tells you exactly which words the model considers urgency signals. This is a direct sanity check on whether the model has learned meaningful patterns or latched onto noise. With a dataset containing synthetic urgent messages built from 55 templates, this interpretability is especially valuable — you can verify the model is picking up on real urgency language rather than memorising template artefacts.

### It generalises well with limited data

With 36 positive examples, generalisation is the primary risk. Logistic Regression with L2 regularisation is one of the most stable classifiers in low-data, high-dimension settings. The regularisation term penalises large weights, preventing the model from over-indexing on rare words that happen to co-occur with urgency labels in training but won't generalise. Ensemble methods and neural networks have far more capacity to overfit at this scale.

### It produces calibrated probabilities

The sigmoid output of Logistic Regression produces well-calibrated probability estimates — `P(urgent) = 0.87` actually means the model is ~87% confident. This matters for a DND bypass system because the decision to interrupt someone is not binary in production: you want a confidence threshold you can tune based on the cost of false positives (unnecessary interruptions) vs false negatives (missed emergencies). Naive Bayes probabilities are poorly calibrated; Logistic Regression's are not.

### It sets a clean performance floor

A strong, well-implemented Logistic Regression baseline gives you a reliable reference point. Any more complex model — Random Forest with metadata, XGBoost, fine-tuned BERT — must beat it to justify the added complexity. If a transformer only marginally outperforms Logistic Regression on this task, that tells you the text signal is largely captured by keyword presence, which is itself a meaningful finding about the nature of urgency language in this dataset.
