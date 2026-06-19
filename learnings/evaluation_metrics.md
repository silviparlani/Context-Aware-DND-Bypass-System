# Evaluation Metrics

---

## The Problem with Accuracy on Imbalanced Data

The DND bypass dataset has 393 non-urgent messages (label `0`) and 36 urgent ones (label `1`). That's roughly a 91:9 split.

The baseline Logistic Regression reports:

```
Accuracy: 0.895
```

That looks good. It isn't. A model that predicts **every single message as non-urgent** would score ~91% accuracy. It would never fire a bypass — the feature is completely broken — and yet it would beat our trained model on this metric.

Accuracy is defined as:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

On imbalanced data, the large TN pool inflates the numerator regardless of how well the model handles the minority class. Accuracy tells you nothing useful here. You need metrics that focus on the minority class in isolation.

---

## The Confusion Matrix

Everything else flows from the confusion matrix. Ours on the test set:

```
[[73  6]
 [ 3  4]]

Layout:
[[TN  FP]
 [FN  TP]]
```

Translating each cell:

| Cell | Count | Meaning |
|---|---|---|
| **TN = 73** | 73 | Non-urgent messages correctly left alone |
| **FP = 6** | 6 | Non-urgent messages that triggered a bypass (false alarm) |
| **FN = 3** | 3 | Urgent messages the model missed entirely |
| **TP = 4** | 4 | Urgent messages correctly caught |

For this project the cost asymmetry matters:
- A **false negative** (FN) means someone in distress sent "Please check on me" and nobody got woken up. That's the dangerous failure.
- A **false positive** (FP) means someone's sleep was interrupted for nothing. That's annoying but not dangerous.

Every metric below is ultimately a different way of aggregating these four numbers.

---

## Precision

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**Question it answers:** Of all the times the model fired the bypass, how often was it actually necessary?

From our results:

```
Class 1 Precision: 0.40
```

The model predicted 10 messages as urgent (4 TP + 6 FP). Only 4 of those were genuinely urgent. So 60% of bypass triggers were false alarms.

**When precision matters most:** When false positives are expensive — spamming users, burning trust, waking someone at 3am for a non-emergency. A precision-focused system is conservative; it only fires when it's very confident.

---

## Recall (Sensitivity)

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**Question it answers:** Of all the genuinely urgent messages, how many did the model actually catch?

From our results:

```
Class 1 Recall: 0.57
```

There were 7 urgent messages in the test set (4 TP + 3 FN). The model found 4 of them — it missed 3. That's a 43% miss rate on the thing that matters most.

**When recall matters most:** When false negatives are dangerous — missed fraud, missed medical alerts, missed emergencies. A recall-focused system is aggressive; it would rather fire unnecessarily than miss a real case.

**The precision-recall tradeoff:** You can always increase recall by lowering the classification threshold (predict urgent if `P(urgent) > 0.3` instead of `> 0.5`). But that comes at a cost — more false positives, so precision drops. There's no free lunch.

---

## F1-Score

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

The F1-score is the harmonic mean of precision and recall. It collapses the tradeoff into a single number that penalises extreme imbalances between the two.

Why harmonic mean instead of arithmetic mean? If precision is 1.0 and recall is 0.0, the arithmetic mean gives 0.5 — which sounds acceptable. The harmonic mean gives 0.0, correctly signalling that the model is useless.

From our results:

```
Class 1 F1-Score: 0.47
```

This is our headline metric for the minority class. With precision at 0.40 and recall at 0.57, the F1 sits closer to the lower value (precision), reflecting that the model is better at finding urgent messages than it is at avoiding false alarms.

**When F1 makes sense:** When you care about both precision and recall and need a single number to compare models or tune thresholds. It's the default choice for imbalanced classification.

---

## The Full Classification Report

```
              precision    recall  f1-score   support

           0       0.96      0.92      0.94        79
           1       0.40      0.57      0.47         7

    accuracy                           0.90        86
   macro avg       0.68      0.75      0.71        86
weighted avg       0.91      0.90      0.90        86
```

Three rows to understand:

**Per-class rows** are the most important. Class `0` has excellent scores across the board because the model has seen hundreds of examples. Class `1` is where the difficulty is — moderate recall (finding over half of urgent messages) but poor precision (too many false alarms).

**Macro avg** averages the metrics treating both classes equally. The macro F1 of 0.71 is dragged down from the near-perfect class-0 score to reflect the weak class-1 performance. Use this when both classes matter equally.

**Weighted avg** averages by support (number of examples per class). Because class `0` has 79 examples vs class `1`'s 7, it dominates. The weighted F1 of 0.90 looks great but is almost entirely driven by class `0`. This number is misleading here — don't use it as the primary metric.

**Rule of thumb:** On imbalanced datasets, always look at per-class metrics and macro averages. Weighted averages and overall accuracy flatly lie.

---

## ROC Curve and AUC

The ROC (Receiver Operating Characteristic) curve sweeps the classification threshold from 0 to 1 and plots, at every threshold:

- **Y-axis — True Positive Rate (TPR / Recall):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — how many actual urgencies were caught
- **X-axis — False Positive Rate (FPR):** $\frac{\text{FP}}{\text{FP} + \text{TN}}$ — how many non-urgencies were falsely flagged

The diagonal baseline (AUC = 0.5) is random guessing. A perfect classifier's curve hugs the top-left corner (AUC = 1.0).

From our results:

```
AUC: 0.76
```

An AUC of 0.76 means: pick a random urgent message and a random non-urgent message — the model ranks the urgent one higher 76% of the time. That's fair performance, clearly better than chance.

**The limitation of ROC-AUC on imbalanced data:** FPR uses TN in its denominator. When TN is very large (as it is here — 73 true negatives), FPR stays low even with several false positives. The ROC curve looks better than it is. On a dataset where the negative class dominates, AUC can be misleading.

---

## Precision-Recall Curve and AUPRC

The Precision-Recall curve is the right tool for imbalanced datasets. It plots:

- **Y-axis — Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$
- **X-axis — Recall:** $\frac{\text{TP}}{\text{TP} + \text{FN}}$

There are no true negatives anywhere in this curve. It focuses entirely on the model's behaviour with respect to the positive class.

A perfect classifier would hold precision at 1.0 across all recall values. In practice, curves slope downward to the right — pushing recall higher (finding more urgent messages) requires lowering the threshold, which lets in more false positives and drops precision.

A useful baseline: the **no-skill classifier** (always predict positive) achieves a precision equal to the class prevalence. For this dataset that's 36/429 ≈ 0.084. Any meaningful model must be substantially above this.

From our results:

```
AUPRC: 0.47
```

This is the Area Under the Precision-Recall Curve. Our model's AUPRC of 0.47 is much higher than the 0.084 baseline, but there's meaningful room to improve. Compare this number across future iterations — it's the most honest single-number summary of how well the model handles the urgent class.

**When to use AUPRC vs AUC-ROC:** Use AUPRC for imbalanced datasets where the positive class is rare. Use AUC-ROC when classes are balanced. When in doubt, look at both.

---

## Failure Case Analysis

Metrics summarise failures; examining them directly tells you *why* they happen.

### False Negatives (missed urgencies)

Messages labelled `1` that the model predicted as `0`:

| Message | Probability |
|---|---|
| "Please check on me" | 0.24 |
| "Please check on me" | 0.24 |
| "Something happened, call me" | 0.23 |

The model assigned these probabilities around 0.23–0.24 — well below the 0.5 threshold. These messages are genuinely ambiguous out of context. "Please check on me" could be casual; without the surrounding conversation or sender metadata, the words alone don't strongly signal urgency. This points toward a future feature: sender identity and contextual history.

### False Positives (unnecessary bypass triggers)

Messages labelled `0` that the model predicted as `1`:

| Message | Probability |
|---|---|
| "Wait I ranted about this to piyush I'll forward those messages" | 0.56 |
| "Wait one sec I'm asking my mom" | 0.58 |
| "Will you pick me up from college?" | 0.55 |
| "okay" | 0.69 |
| "Two...i need to take a bath rn cause i have a meet after that" | 0.53 |
| "Okay" | 0.69 |

Two patterns stand out:

1. **"okay" / "Okay"** scoring 0.69 — this is suspicious. The model has overfit to something about short affirmative messages that co-occurred with urgent labels in training. This could be a data artifact from how the urgent templates were written.
2. **Messages with help-adjacent language** ("asking my mom", "pick me up") — the model is pattern-matching on surface-level request language without understanding that these are mundane asks.

Both failure modes point at the same root cause: TF-IDF has no context. It sees isolated words and scores based on their training-set associations. A model with access to conversation history, sender metadata, or contextual embeddings would handle these cases differently.

---

## Which Metrics to Watch for This Project

Given the DND bypass context (imbalanced, high cost of false negatives):

| Metric | Priority | Why |
|---|---|---|
| **Recall (class 1)** | Highest | Missing an urgent message is the worst outcome |
| **F1 (class 1)** | High | Balances recall against false alarm rate |
| **AUPRC** | High | Honest summary metric for imbalanced positive class |
| **Precision (class 1)** | Medium | Unnecessary interruptions degrade user trust |
| **AUC-ROC** | Medium | Useful cross-model comparison, but optimistic here |
| **Overall Accuracy** | Low | Misleading on this class distribution |
| **Weighted avg F1** | Low | Dominated by majority class, hides minority-class failures |

The practical goal: improve recall on class `1` without letting precision collapse entirely. The current 0.57 recall means roughly 4 in 10 urgent messages go undetected. That's the number to move.
