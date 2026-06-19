# Decision Engine — Threshold Tuning

**Date:** 2026-06-04
**Notebook:** `notebooks/DND_Bypass_DecisionEngine.ipynb`
**Builds on:** [`decision_engine.md`](decision_engine.md)

Replaced the initial hand-picked thresholds (0.85 / 0.65) with empirically grounded ones derived from probability distributions and a precision-recall sweep on the full labeled set.

---

## What changed vs the previous decision-engine cut

| | Previous (2026-06-03) | Current (2026-06-04) |
|---|---|---|
| Eval set | 100-row random sample of `dnd_df` | **Full 454-row labeled set** |
| Thresholds | Hand-picked: BYPASS ≥0.85, SILENT_SUMMARY ≥0.65 | **Empirically derived: BYPASS ≥0.65, SILENT_SUMMARY ≥0.40** |
| Threshold justification | None | Probability distribution + PR sweep |
| FP / FN @ initial thresholds | 0 / 1 | 0 / 10 (problem more visible on full set) |

The scorer (`score_message`) is unchanged. The notebook re-runs the initial policy on the full set first — which makes the conservatism much more visible (10 false negatives, not 1) — then derives new thresholds from the data.

---

## Re-evaluation at original thresholds (full set, n=454)

| Action | Count |
|---|--:|
| SUPPRESS | 422 |
| BYPASS_DND | 20 |
| SILENT_SUMMARY | 12 |

| label \ action | BYPASS_DND | SILENT_SUMMARY | SUPPRESS | All |
|---|--:|--:|--:|--:|
| 0 | 0 | 0 | 412 | 412 |
| 1 | 20 | 12 | **10** | 42 |
| **All** | 20 | 12 | 422 | 454 |

- **False positives: 0** — confirms there's headroom to drop thresholds.
- **False negatives: 10** — and every one of them is obviously urgent: *"URGENT"* (0.625), *"SOS"* (0.524), *"This can't wait, call me now"* (0.548 / 0.463), *"Please tell me you're okay"* (0.594), *"I'm not okay right now"* (0.561), etc. The 0.65 SILENT_SUMMARY floor was cutting straight through real urgency.

---

## Probability distribution analysis

`decision_results.groupby("label")["probability"].describe()`:

| label | mean | std | min | 25% | 50% | 75% | max |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0.141 | 0.069 | 0.026 | 0.098 | 0.120 | 0.168 | **0.640** |
| 1 | 0.769 | 0.179 | 0.206 | **0.664** | 0.840 | 0.891 | 0.943 |

The classes separate at **non-urgent max = 0.640** vs **urgent Q1 = 0.664** — a clean ~0.64–0.66 boundary with almost no overlap. This is the natural BYPASS threshold; everything above 0.65 is empirically urgent on this set.

---

## Threshold sweep (single-cutoff binary view)

| threshold | precision | recall | F1 |
|--:|--:|--:|--:|
| 0.30 | 0.678 | 0.952 | 0.792 |
| 0.35 | 0.851 | 0.952 | 0.899 |
| **0.40** | **0.976** | **0.952** | **0.964** |
| 0.45 | 0.976 | 0.952 | 0.964 |
| 0.50 | 0.975 | 0.929 | 0.951 |
| 0.65 | 1.000 | 0.762 | 0.865 |
| 0.85 | 1.000 | 0.476 | 0.645 |

F1 peaks at **threshold 0.40** (F1 = 0.964, precision 0.976, recall 0.952). Anything below 0.40 is high-confidence non-urgent; anything above 0.65 is high-confidence urgent. The 0.40–0.65 band is the genuinely ambiguous middle.

---

## Final thresholds

| Action | Range | Rationale |
|---|---|---|
| SUPPRESS | p < 0.40 | F1-optimal cutoff from the sweep; clear non-urgent zone |
| SILENT_SUMMARY | 0.40 ≤ p < 0.65 | Ambiguous middle — surface without interrupting |
| BYPASS_DND | p ≥ 0.65 | Above the max non-urgent probability (0.640) and near urgent-Q1 (0.664) |

Compared to the previous policy: the BYPASS threshold drops from 0.85 → 0.65, and the SILENT_SUMMARY band moves from 0.65–0.85 (where it was acting as a soft-reject bucket for urgent messages) to 0.40–0.65 (its intended role: genuinely uncertain middle).

---

## Findings

### 1. Wider eval surfaces the real problem
The 100-row sample on 2026-06-03 only revealed one false negative; the full 454-row set revealed ten. Sample-size dependence of FN counts at low prevalence (~9% urgent) makes small-sample evals misleading for threshold work.

### 2. Class probabilities are well-separated
Non-urgent max 0.640 vs urgent Q1 0.664 means the RF is producing genuinely separable scores — the prior conservatism was a policy problem, not a model problem.

### 3. SILENT_SUMMARY now has a coherent purpose
Previously it caught messages the policy was *too scared to bypass*; now it catches messages the model is *genuinely uncertain about*.

---

## Next Steps

- Validate on a true held-out split (the threshold sweep above was on the same labeled set the RF was trained on — risk of optimism).
- Inspect the urgent message with p = 0.206 (`"Please"`) — the lone urgent outlier far below the SILENT_SUMMARY floor.
- Re-run the engine on `data/chats_with_contextual_features.csv` (the unlabeled full set) to see action distribution at production scale.
- Package `score_message` + `notification_action` (with the new thresholds) into `src/`.
