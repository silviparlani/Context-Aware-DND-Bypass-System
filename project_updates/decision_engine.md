# Decision Engine — From Classifier to Notification Policy

**Date:** 2026-06-03
**Notebook:** `notebooks/DND_Bypass_DecisionEngine.ipynb`
**Artifacts:** `dnd_df.pkl`, `dnd_rf_model.pkl`, `dnd_vectorizer.pkl`, `metadata_features.pkl`

First end-to-end notification policy on top of the trained Random Forest (see [`random_forest.md`](random_forest.md)). The classifier becomes a deployable two-stage engine: **scorer** (hybrid features → urgency probability) + **policy** (probability → notification action).

---

## Scorer — `score_message()`

Reconstructs the hybrid feature pipeline at inference time:

- `vectorizer.transform([message])` → 1×732 TF-IDF row
- `csr_matrix([[sender_importance, is_group_chat, hour_of_day, average_messages_per_day]])` → 1×4 metadata row
- `hstack` → 1×736 sparse row matching the training matrix
- `model.predict_proba(...)[0][1]` → P(urgent)

Smoke test: `("Call me ASAP", sender_importance=3, is_group_chat=0, hour_of_day=2, average_messages_per_day=1)` → **0.660**.

---

## Policy — `notification_action(prob)`

| Probability | Action | Intent |
|---|---|---|
| ≥ 0.85 | `BYPASS_DND` | Interrupt — high-confidence urgent |
| ≥ 0.65 | `SILENT_SUMMARY` | Surface without sound — borderline urgent |
| < 0.65 | `SUPPRESS` | Hold for normal review |

Three-tier policy (rather than binary urgent/not-urgent) to give the borderline band a softer landing than a full DND bypass.

---

## Evaluation — 100-row random sample (`random_state=42`)

Action distribution:

| Action | Count |
|---|--:|
| SUPPRESS | 94 |
| SILENT_SUMMARY | 3 |
| BYPASS_DND | 3 |

Action × true label:

| label \ action | BYPASS_DND | SILENT_SUMMARY | SUPPRESS | All |
|---|--:|--:|--:|--:|
| 0 (non-urgent) | 0 | 0 | 93 | 93 |
| 1 (urgent) | 3 | 3 | 1 | 7 |
| **All** | 3 | 3 | 94 | 100 |

- **False positives: 0** — no non-urgent message escaped to either notify tier.
- **False negatives: 1** — *"This is not a joke, call me"* (p = 0.636) fell just below the 0.65 threshold and was suppressed.

BYPASS_DND examples (all correct): *"Are you safe?"* (0.91), *"URGENT"* (0.85), *"I've been calling you for an hour"* (0.91).

SILENT_SUMMARY examples (all label `1`): *"Anyone reachable? bua is unwell, need someone now"* (0.85), *"Not safe right now"* (0.66), *"Please pick up, it's important"* (0.81).

---

## Findings

### 1. Policy is too conservative
All three SILENT_SUMMARY messages are unambiguously urgent — they should have bypassed DND outright. The 0.85 BYPASS threshold is leaving real interrupts on the table.

### 2. The SILENT_SUMMARY band is doing the wrong job
As tuned, it behaves as a soft-reject bucket for urgent messages, not as a band for genuinely borderline cases. Either the threshold needs to drop or the tier needs a different purpose.

### 3. The one false negative is on the threshold edge
*"This is not a joke, call me"* scored 0.636 — within 0.014 of SILENT_SUMMARY. A modest threshold drop recovers this case without obviously breaking precision (FPs were 0 on this sample).

### 4. Precision is the easy half
Zero false positives across 93 non-urgent messages suggests headroom to trade some precision for recall by lowering thresholds.

---

## Next Steps

- Sweep the (BYPASS, SILENT_SUMMARY) threshold pair on the full set; plot action-mix and FP/FN counts to pick a less conservative operating point.
- Re-evaluate on the held-out test split from the RF notebook (not just a fresh 100-row sample of the training pool) for an honest read.
- Decide what SILENT_SUMMARY is *for* — borderline-urgent triage, or a low-noise channel for medium-confidence cases — and tune thresholds to that intent.
- Package `score_message` + `notification_action` into `src/` so the engine is importable outside the notebook.
