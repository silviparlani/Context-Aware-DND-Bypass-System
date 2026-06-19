# Contextual Feature Engineering

**Date:** 2026-05-17
**Notebook:** `notebooks/DND_Bypass_ContextualFeatureEngineering.ipynb`
**Dataset:** `data/chats_sample_features.csv` (525 rows; label `-1` rows excluded → 454 rows for feature analysis)

---

## Dataset Setup

- Froze the labeled baseline sample as `chats_sample_baseline.csv` (untouched, 500 rows).
- Created `chats_sample_features.csv` as the working copy for feature engineering.
- Seeded 5 synthetic group senders into `chats_clean.csv` (50 rows) and `chats_sample_features.csv` (25 rows) to expose the model to group-chat behavior:

| Sender | Priority | Profile |
|---|:-:|---|
| `group_1` | 5 | Family emergency group |
| `group_2` | 4 | Work / team channel |
| `group_3` | 3 | Close friends |
| `group_4` | 2 | Society / neighbours |
| `group_5` | 1 | Social / memes |

---

## Feature 1 — `sender_importance`

- Manual mapping from each sender (Person_1..Person_14, group_1..group_5) to an integer 1–5.
- Mapping derived from per-sender message-tone analysis: crisis/urgency keyword density, casual/media-omitted ratio, and qualitative review of sampled messages.
- High-priority (5): Person_5–10, group_1. Low-priority (1): Person_4, Person_11, Person_12, group_5.

**Findings:**
- Mean `sender_importance` for urgent (label 1): **3.74**
- Mean `sender_importance` for non-urgent (label 0): **1.99**
- **~1.75-point separation** between classes — the strongest contextual signal so far and likely the single best non-text feature.

---

## Feature 2 — `hour_of_day` and `day_of_week`

- Parsed `timestamp` to datetime (with fallback parsing from the `date` + `time` columns for rows where the primary parse fails).
- Extracted `hour_of_day` (0–23) and `day_of_week` (Monday..Sunday).

**Findings:**
- Mean `hour_of_day` for urgent: **10.83**
- Mean `hour_of_day` for non-urgent: **14.90**
- Urgent messages skew earlier in the day, consistent with EDA's bimodal urgent peaks (~3 AM and ~1 PM). Signal is real but weak as a single dimension — overlaps heavily with non-urgent messaging hours.
- `day_of_week` not yet evaluated quantitatively against the label.

---

## Feature 3 — `is_group_chat`

- Binary flag: 1 if sender starts with `group_`, else 0.

**Findings:**

| | label 0 | label 1 |
|---|--:|--:|
| **Non-group (0)** | 393 | 36 |
| **Group (1)** | 19 | 6 |

- Group senders show a higher proportion of urgent messages (6/25 = 24%) than non-group (36/429 = 8.4%), but the absolute volume is small.
- Moderate signal — useful, realistic, worth keeping.

---

## Feature 4 — `average_messages_per_day`

- Grouped by `(sender, date)` to get daily message counts, then averaged over all active days per sender.
- Joined back onto every row.

**Findings:**
- Mean for urgent: **1.04**
- Mean for non-urgent: **1.14**
- Very small spread (full range ≈ 1.0–1.18). Weak signal on its own — high-frequency senders aren't reliably non-urgent in this sample, and most senders cluster near 1 message/day after the dedup-style daily aggregation. May still contribute marginally in combination.

---

## Feature 5 — `recent_message_frequency`

- 10-minute rolling count of messages per sender (`groupby('sender').rolling('10min').count()`).

**Findings:**
- 440 of 454 rows = 1.0; only 14 rows = 2.0.
- Distribution is far too sparse on this sample to discriminate — the time window may need widening, or this feature is more useful at inference time on live message streams than on the static historical sample.

---

## Feature Signal Summary

| Feature | Strength | Notes |
|---|:-:|---|
| `sender_importance` | **Strong** | Largest class separation; best non-text feature so far |
| `is_group_chat` | Medium | Realistic, modest discriminative power |
| `hour_of_day` | Weak (alone) | Echoes EDA; useful combined with sender |
| `average_messages_per_day` | Weak | Near-uniform on this sample |
| `day_of_week` | TBD | Not yet evaluated vs label |
| `recent_message_frequency` | Weak (this sample) | Too sparse on historical data; reconsider at inference |

---

## Next Steps

- Retrain the baseline classifier on `chats_sample_features.csv` augmented with the new feature columns and compare class-1 F1 / AUPRC against the text-only baseline (F1 0.47, AUPRC 0.47).
- Consider widening the rolling window for `recent_message_frequency`, or replacing it with time-since-last-message-from-sender.
- Run a quick mutual-information / feature-importance pass to confirm the qualitative signal ranking above.
