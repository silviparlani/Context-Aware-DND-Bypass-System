# Baseline Model & Evaluation

**Date:** 2026-05-16  
**Notebook:** `notebooks/DND_Bypass_BaselineModel.ipynb`  
**Dataset:** `data/chats_sample_baseline.csv` — label `-1` rows dropped; 393 class-0, 36 class-1

---

## Setup

- **Train/test split:** 80/20, `stratify=y`, `random_state=42` — preserves the class imbalance ratio in both sets.
- **Vectorization:** TF-IDF, `stop_words='english'`, `max_features=3000`. Fit on training set only; test set is transform-only to prevent data leakage.
- **Model:** Logistic Regression, `class_weight='balanced'`, `solver='liblinear'`, `max_iter=1000`.

---

## Results

| Metric | Class 0 | Class 1 |
|---|---|---|
| Precision | 0.96 | 0.40 |
| Recall | 0.92 | 0.57 |
| F1-score | 0.94 | 0.47 |
| Support | 79 | 7 |

**Overall accuracy:** 0.895  
**AUC-ROC:** 0.76  
**AUPRC:** 0.47

**Confusion matrix:**

```
[[73  6]
 [ 3  4]]
 
 TN=73  FP=6
 FN=3   TP=4
```

---

## Model Performance Observations

- Accuracy of ~90% is misleading in isolation. On a 91:9 imbalanced dataset, a model that predicts every message as non-urgent would score ~91% accuracy while being completely useless. The relevant metric is performance on class 1.

- The recall of 0.57 means the model identified just over half of all urgent messages. Given that the primary failure mode for this system is a missed emergency, this is the number to improve most urgently.

- The precision of 0.40 means 60% of bypass triggers were false alarms — 6 unnecessary interruptions in a test set of 86 messages. This is tolerable at baseline but would degrade user trust at scale.

- The model produces more false positives than false negatives (6 vs 3), which is the preferable failure direction for this use case: an unnecessary interruption is annoying, but a missed emergency can be dangerous.

- The class-1 F1 of 0.47 reflects a model that is finding some signal but is not yet reliable. It is a reference point, not a target.

---

## ROC-AUC & Precision-Recall Analysis

- The AUC-ROC of 0.76 confirms the model learned real patterns — it ranks a randomly selected urgent message above a randomly selected non-urgent one 76% of the time. This is meaningfully better than chance and validates that urgency carries detectable linguistic signals even in a bag-of-words representation.

- However, AUC-ROC is an optimistic metric on imbalanced data. Because the False Positive Rate uses True Negatives in its denominator (FPR = FP / (FP + TN)), a large TN pool keeps FPR low even with several false positives. The ROC curve looks better than the PR curve for this reason.

- The Precision-Recall curve tells the more honest story. The AUPRC of 0.47 is far above the no-skill baseline (~0.084 = class prevalence), but the rapid precision drop as recall increases shows how quickly the model starts producing false alarms when pushed to catch more urgent messages.

- The jagged shape of the PR curve is partly a function of the small test set (only 7 urgent messages) — each individual prediction has an outsized effect on the curve.

- **Primary metric for future iterations:** AUPRC. It is the most honest single-number summary of performance on the positive class for an imbalanced dataset.

---

## False Negative Analysis (Missed Urgent Messages)

Messages labelled urgent that the model missed:

| Message | P(urgent) |
|---|---|
| "Please check on me" | 0.24 |
| "Please check on me" | 0.24 |
| "Something happened, call me" | 0.23 |

**Pattern:** All three are short, vague, and context-dependent. The urgency is implicit — it depends on knowing the sender, the relationship, or the surrounding conversation. Without that context, the words themselves are unremarkable.

**Linguistic ambiguity:** Words like "please", "check", "call", and "happened" appear in both urgent and non-urgent messages throughout the dataset. They carry statistical weight but not enough to push the model past the 0.5 threshold. TF-IDF cannot distinguish "please check on me" (distress) from "please check the document" (routine request).

**Short message limitation:** Very short messages produce sparse TF-IDF vectors. With few active features, the prediction becomes unstable and defaults toward the majority class. Bag-of-words methods are structurally disadvantaged when the message provides minimal text signal.

**Implication:** Urgency in this dataset is often communicated implicitly, relying on contextual cues that text alone cannot capture. Sender identity, repeated messaging behavior, and time-of-day are likely necessary to catch these cases.

---

## False Positive Analysis (Unnecessary Bypass Triggers)

Messages labelled non-urgent that the model incorrectly flagged:

| Message | P(urgent) |
|---|---|
| "Wait I ranted about this to piyush I'll forward those messages" | 0.56 |
| "Wait one sec I'm asking my mom" | 0.58 |
| "Will you pick me up from college?" | 0.55 |
| "okay" | 0.69 |
| "Two...i need to take a bath rn cause i have a meet after that" | 0.53 |
| "Okay" | 0.69 |

**Rare token sensitivity:** TF-IDF assigns high importance to words that appear infrequently across the corpus. Informal tokens like "ranted", "rn", "sec", and "bath" may have co-occurred with urgent messages enough times in training to inflate their weight. The model treats statistical rarity as a proxy for importance, which fails for casual conversational language.

**Short non-urgent messages:** "okay" and "Okay" scoring 0.69 is a red flag — these are almost certainly data artifacts. Single-word messages provide almost no signal, so the prediction is driven by weak statistical associations from training. The urgency templates likely ended with or prompted short confirmatory replies in context.

**Subjective urgency:** "Will you pick me up from college?" is a genuine edge case. Depending on the sender and circumstances, this could be routine or urgent. The model cannot access that context and makes the only decision available to it: classify based on surface language.

**Implication:** False positives are not random noise — they cluster around two failure modes: (1) rare token overfitting, and (2) missing social/contextual context. Both point toward the same solution as the false negatives: richer features beyond text.

---

## Broader Interpretation

- The baseline demonstrates that NLP-based notification prioritization is feasible with simple techniques. A TF-IDF + Logistic Regression model, trained on fewer than 350 examples, achieves above-chance discrimination on a genuinely difficult task.

- The failure cases make the path forward legible. Both false negatives and false positives stem from the same root cause: urgency is often socially and contextually determined, not linguistically explicit. Text-only models hit a ceiling on this task.

- The next phase should introduce metadata features — sender identity, time of day, message frequency, and repeated-message behavior — alongside the text signal. Tree-based models (Random Forest, XGBoost) are the right tool for that feature mix. See `learnings/baseline_models.md` for the rationale.
