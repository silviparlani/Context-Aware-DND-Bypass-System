# Random Forest — Non-Linear Hybrid Model

**Date:** 2026-05-20
**Notebook:** `notebooks/DND_Bypass_ContextualFeatureEngineering.ipynb`
**Dataset:** `data/chats_sample_features.csv` (454 rows after excluding label `-1`)

Second pass on the hybrid feature matrix `X_combined` (454 × 736 — see [`hybrid_model.md`](hybrid_model.md)). Same train/test split as the Logistic Regression baseline (`test_size=0.2`, `random_state=42`, `stratify=y`) for a like-for-like comparison.

---

## Model

```
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    random_state=42
)
```

- `class_weight='balanced'` to compensate for the ~8% urgent prevalence.
- No further tuning — first cut, identical features to the LR hybrid model.

---

## Results — Class `1` (Urgent)

| Metric | LR Hybrid | Random Forest | Δ |
|---|--:|--:|--:|
| Accuracy | 0.96 | **0.97** | +0.01 |
| Precision | 0.83 | **0.86** | +0.03 |
| Recall | 0.62 | **0.75** | **+0.13** |
| F1 | 0.71 | **0.80** | **+0.09** |

| Errors (test set, n=91) | LR Hybrid | Random Forest |
|---|--:|--:|
| False Positives | 1 | 1 |
| False Negatives | 3 | **2** |

Confusion matrix:
```
[[82  1]   (non-urgent: 82 TN, 1 FP)
 [ 2  6]]  (urgent:      2 FN, 6 TP)
```

Recall improved with no extra false positives — Random Forest captured contextual interactions more effectively without sacrificing reliability.

---

## Observations

### 1. Random Forest improved urgent message detection
Recall **0.62 → 0.75** — better identification of urgent messages.

### 2. Precision remained stable
The model maintained high urgent-class precision (0.83 → 0.86) while improving recall, avoiding any increase in notification noise.

### 3. False negatives decreased
**3 → 2** — improved capture of contextually urgent messages that LR was missing.

### 4. Contextual interactions appear important
Results suggest urgency emerges from *interactions* between message text, sender importance, timing, and communication behaviour — rather than any one feature in isolation. LR (additive log-odds) can't represent those interactions; RF can.

### 5. Tree-based models better captured behavioural context
Compared to Logistic Regression, Random Forest appeared more capable of modelling non-linear contextual urgency patterns — consistent with the recall lift on the same feature matrix.

---

## Visualizations

Three comparison charts added to the notebook:

1. **LR vs RF — extended metrics** (Precision / Recall / F1 / Accuracy)
2. **LR vs RF — error comparison** (False Positives / False Negatives)
3. **LR vs RF — urgent-class performance** (combined view)

---

## Next Steps

- Inspect RF feature importances — confirm whether `sender_importance` still dominates and how the TF-IDF tokens rank against the four metadata features.
- Try gradient boosting (XGBoost / LightGBM) as a stronger non-linear baseline; compare against the RF.
- Threshold sweep / precision-recall curve on the RF probabilities to see whether further recall is reachable.
- Revisit the `sender_importance` leakage concern (manual mapping derived from per-sender tone on the same sample) — a held-out or cross-validated importance score would be a cleaner test now that RF is leaning on it.
