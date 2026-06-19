# Hybrid Model — Text + Contextual Metadata

**Date:** 2026-05-19
**Notebook:** `notebooks/DND_Bypass_ContextualFeatureEngineering.ipynb`
**Dataset:** `data/chats_sample_features.csv` (454 rows after excluding label `-1`)

This is the project's **first hybrid NLP + metadata architecture** — the transition from pure NLP classification to multi-signal contextual decision modeling, and the first context-aware urgency classifier.

---

## Feature Assembly

- **Text features** — `TfidfVectorizer(stop_words='english', max_features=3000)` over the `message` column → `X_text`.
- **Metadata features** — `['sender_importance', 'is_group_chat', 'hour_of_day', 'average_messages_per_day']` → `X_meta`, converted to `csr_matrix` as `X_meta_sparse`.
- **Combined matrix** — `hstack([X_text, X_meta_sparse])` → `X_combined` with shape **(454, 736)** and 2,547 stored non-zero elements.
- `day_of_week` and `recent_message_frequency` deliberately excluded (the former not yet quantitatively validated; the latter too sparse on this sample — see [`feature_engineering.md`](feature_engineering.md)).

---

## Train / Test Split

- `train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)`.
- Stratification preserves the ~8% urgent prevalence in both folds.

---

## Model

- `LogisticRegression(max_iter=1000)` fit on `X_combined`.
- No class weighting, no regularization tuning — first cut, intentional like-for-like comparison with the text-only baseline.

---

## Results

| Metric | Baseline (text-only) | Hybrid (text + metadata) | Δ |
|---|--:|--:|--:|
| Accuracy (overall) | 89.5% | **96%** | **+6.5 pts** |
| Precision (urgent) | 0.40 | **0.83** | **+0.43** |
| Recall (urgent) | 0.57 | 0.62 | +0.05 |
| F1 (urgent) | 0.47 | **0.71** | **+0.24** |

| Errors (test set) | Baseline | Hybrid |
|---|--:|--:|
| False Positives | 6 | **1** |
| False Negatives | 3 | 3 |

The lift is overwhelmingly driven by precision — false positives collapsed 6 → 1 while recall held. The hybrid model interrupts DND far less often without missing more urgent messages.

**What this means:**
- **Improved notification reliability.** Urgent predictions are now substantially more trustworthy (precision 0.40 → 0.83) — fewer unnecessary DND interruptions.
- **Recall improved modestly** (0.57 → 0.62). Contextual signals help detect some additional urgent messages, but the primary benefit is suppressing false alarms.

---

## Error Analysis

Test indices were preserved on a second split so predictions could be joined back to the original rows for inspection.

### False Positives (n=1)
- Single FP appeared on an emotionally-charged message from a high-importance sender. The model is correctly weighting `sender_importance` heavily; the failure mode is essentially a label-boundary case rather than a model defect.

### False Negatives (n=3) — observations
1. **Vague emotional language is hard.** Messages like *"I'm not okay right now"* express distress without explicit urgency keywords or action requests — TF-IDF has no hook to latch onto.
2. **Metadata can override strong text signals.** *"This can't wait, call me now"* was classified non-urgent — strong text cues were suppressed by a low-priority sender importance.
3. **`sender_importance` is highly influential.** Multiple FNs involved low-importance senders. The model leans on trusted-contact priors more than on textual urgency.
4. **Temporal signals are weak alone.** Some FNs occurred at 3–4 AM; `hour_of_day` did not compensate for low sender importance.
5. **Administrative content dilutes urgency.** Long structured messages (phone numbers, contact details, repeated info) statistically overshadow short urgent phrases inside the TF-IDF space.
6. **Urgency is subjective.** Some "urgent-sounding" FNs probably wouldn't warrant a real DND interruption — a labelling-philosophy issue, not a modelling one.

### Tradeoff — Reducing Notification Noise vs Maximizing Urgent Recall
The hybrid model trades aggressive urgency detection for far better notification reliability. The core system tension is now explicit:

- **Minimize unnecessary interruptions** (favoured by contextual metadata — trust signals suppress urgent predictions from low-priority senders).
- **Maximize urgent message recall** (favoured by raw text cues — but those cues alone over-fire).

Whether the current balance is right is a product decision, not a metric one — but for the personal-DND use case it's almost certainly correct.

---

## Visualizations

The notebook produced two comparison charts intended for README / portfolio presentation:

1. **Baseline vs Hybrid metric comparison** — grouped bar chart of Precision / Recall / F1 on the urgent class.
2. **Error comparison** — grouped bar chart of False Positives and False Negatives, baseline vs hybrid.

Both visualizations make the precision-driven story legible at a glance and are the cleanest single-frame summary of the contextual-features payoff.

---

## Next Steps

- Tune the precision/recall balance — class-weighted logistic regression or threshold sweep — to see if recall can be lifted without giving back the FP gains.
- Re-introduce `day_of_week` and a wider-window `recent_message_frequency`; verify with mutual information before adding to the model.
- Try a non-linear model (gradient-boosted trees) on the same `X_combined` to check whether the text/metadata interactions Logistic Regression can't capture are leaving signal on the table.
- Consider sender-importance leakage: the manual 1–5 mapping was derived from per-sender tone analysis on the same sample. A held-out sender or cross-validated importance score would be a cleaner test.
