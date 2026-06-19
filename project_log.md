# DND_Bypass — Project Log

---

## Phase 1 — Data Collection & Preprocessing

Exported 4 WhatsApp chats (~85k messages), augmented with ~8.5k synthetic urgent messages, anonymized, parsed to CSV, cleaned of junk rows, sampled to 500 rows, and labeled. Working dataset: `chats_sample.csv` — 500 rows, 36 urgent (label `1`), 394 normal (label `0`), 70 non-English/content-free (label `-1`).

→ `project_updates/preprocessing.md`

---

## 2026-05-05 — Exploratory Data Analysis

EDA on the 500-row sample. Covered time-of-day distribution, sender urgency proportions, message frequency heatmaps (all + urgent-only), and message length distributions. Key finding: sender identity and time-of-day are strong signals; message text alone will not be sufficient.

→ `project_updates/eda.md`

---

## 2026-05-16 — Baseline Model & Evaluation

Logistic Regression + TF-IDF baseline. Class-1 results: precision 0.40, recall 0.57, F1 0.47, AUC-ROC 0.76, AUPRC 0.47. False negatives were short, context-dependent messages with implicit urgency. False positives were driven by rare tokens and conversational ambiguity. Text alone is insufficient — metadata features are the clear next step.

→ `project_updates/baseline_model.md`

---

## 2026-05-17 — Contextual Feature Engineering

Froze the baseline sample as `chats_sample_baseline.csv` and forked a working copy, `chats_sample_features.csv`, for feature work. Seeded 5 synthetic group senders (`group_1`..`group_5`, priorities 5→1) into `chats_clean.csv` and the features sample to give the model exposure to group-chat behavior. Engineered six contextual features on the features sample: `sender_importance` (1–5 mapping derived from per-sender message tone), `hour_of_day`, `day_of_week`, `is_group_chat`, `average_messages_per_day`, and `recent_message_frequency` (10-min rolling count). `sender_importance` is the strongest signal (mean 3.74 for urgent vs 1.99 for non-urgent); `is_group_chat` is a moderate signal; `hour_of_day` and `average_messages_per_day` are weak in isolation but expected to add value combined.

→ `project_updates/feature_engineering.md`

---

## 2026-05-19 — Hybrid Model (Text + Metadata)

First hybrid NLP + metadata architecture — the project's transition from pure NLP classification to multi-signal contextual decision modeling. Combined the TF-IDF text vector (`max_features=3000`) with four metadata columns (`sender_importance`, `is_group_chat`, `hour_of_day`, `average_messages_per_day`) via `scipy.sparse.hstack` into a 454×736 sparse matrix, and trained Logistic Regression on an 80/20 stratified split. Accuracy **89.5% → 96%**; class-1 precision **0.40 → 0.83**, recall **0.57 → 0.62**, F1 **0.47 → 0.71**. False positives collapsed 6 → 1 with no extra false negatives — the hybrid model interrupts DND far less often without missing more urgent messages. Error analysis confirmed `sender_importance` dominates: strong text cues from low-priority senders were suppressed (e.g. *"This can't wait, call me now"* → non-urgent), and vague emotional language without explicit urgency keywords (e.g. *"I'm not okay right now"*) remains hard. Produced baseline-vs-hybrid metric and error-comparison charts for README / portfolio use.

→ `project_updates/hybrid_model.md`

---

## 2026-05-20 — Random Forest on Hybrid Features

Trained `RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42)` on the same `X_combined` matrix and the same 80/20 stratified split. Class-1 metrics: accuracy **0.96 → 0.97**, precision **0.83 → 0.86**, recall **0.62 → 0.75**, F1 **0.71 → 0.80**; false negatives **3 → 2** with false positives unchanged at 1. Recall lifted without giving back precision — Random Forest captured non-linear interactions between text, sender importance, timing, and communication behaviour that Logistic Regression's additive log-odds couldn't represent. Added LR-vs-RF comparison table and three bar charts (extended metrics, error comparison, urgent-class performance).

→ `project_updates/random_forest.md`

---

## 2026-06-03 — Decision Engine (Scorer + Policy)

Wrapped the trained Random Forest into a deployable two-stage notification engine. `score_message(message, sender_importance, is_group_chat, hour_of_day, average_messages_per_day)` reconstructs the 1×736 hybrid TF-IDF + metadata row at inference time and returns `predict_proba` for class 1; `notification_action(prob)` maps the probability to one of three actions — **BYPASS_DND** (≥ 0.85), **SILENT_SUMMARY** (≥ 0.65), or **SUPPRESS**. Smoke test on 100 random rows (`random_state=42`): 3 BYPASS_DND, 3 SILENT_SUMMARY, 94 SUPPRESS, with **0 false positives** and **1 false negative** (*"This is not a joke, call me"*, p = 0.636, just below the SILENT_SUMMARY threshold). Policy is clearly too conservative — all three SILENT_SUMMARY messages were unambiguously urgent and should have bypassed DND. Thresholds need re-tuning, and the SILENT_SUMMARY tier needs a clearer purpose. First step toward turning the classifier into a usable notification product.

→ `project_updates/decision_engine.md`

---

## 2026-06-04 — Decision Engine: Empirical Threshold Tuning

Replaced the hand-picked 0.85 / 0.65 thresholds with empirically derived ones. Re-ran the original policy on the **full 454-row labeled set** instead of a 100-row sample, which surfaced **10 false negatives** (vs 1 on the sample) — including *"URGENT"* (p=0.625), *"SOS"* (0.524), and *"This can't wait, call me now"* (0.548). Probability distributions are cleanly separated: non-urgent max **0.640**, urgent Q1 **0.664** — a natural class boundary the original BYPASS threshold (0.85) was sitting far above. Threshold sweep (binary view, 0.10–0.95) peaks at F1 = **0.964** at threshold **0.40** (precision 0.976, recall 0.952). New boundaries: **SUPPRESS < 0.40**, **SILENT_SUMMARY 0.40–0.65**, **BYPASS ≥ 0.65** — BYPASS floor dropped 0.85 → 0.65, and SILENT_SUMMARY moved from a soft-reject bucket (0.65–0.85) to its intended ambiguous-middle role (0.40–0.65). Caveat: thresholds were tuned on the same labeled set the RF was trained on — held-out validation is the next step.

→ `project_updates/decision_engine_thresholds.md`

---

## 2026-06-11 — Deployment: Module Extraction + FastAPI Service

Bridged the Week 5 notebook into a deployable service. Extracted the production code from `DND_Bypass_NotificationSimulation.ipynb` into `app/predictor.py` (artifact loading + `score_message`, `notification_action`, `process_notification`, and the action routers), using file-relative model paths so it runs from any working directory. Stood up a fresh project venv (`.venv/`) pinned to **scikit-learn 1.6.1** to match the model pickles, captured in `requirements.txt`. Added a 31-test pytest suite (`tests/`, `requirements-dev.txt`): the headline regression test replays the notebook's exact 11-message simulation through `predictor.py` and asserts every probability matches the original table to 3 decimals — proving the module is a faithful copy, not just runnable — alongside threshold-boundary, output-schema, and action-routing tests. **All 31 pass.** Wrapped the predictor in a FastAPI service (`app/main.py`) exposing `POST /predict`; verified end-to-end via `/docs` and live calls (e.g. *"Dad is in the hospital"* → p=0.786 → **BYPASS_DND**), with auto-generated OpenAPI docs. Produced a portfolio architecture diagram (`images/architecture.png`, regenerable via `images/make_architecture.py`) showing the full pipeline from incoming notification → features → Random Forest → decision policy → `/predict`. First deployable version of the system. Known follow-up: `/predict` accepts a raw `dict`, so missing fields 500 rather than returning a clean 422 — swap to a Pydantic model during README/polish.

→ `project_updates/deployment.md`
