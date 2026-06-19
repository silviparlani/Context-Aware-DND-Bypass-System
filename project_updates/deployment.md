# Deployment — Module Extraction + FastAPI Service

**Date:** 2026-06-11
**Phase:** Week 6 — Deployment & Polish

## Goal

Turn the Week 5 decision engine (living inside a notebook) into a deployable
service that can be called over HTTP, without changing the model or the decision
logic.

## What was done

### 1. Module extraction — `app/predictor.py`
Moved the production code out of `DND_Bypass_NotificationSimulation.ipynb`:
- Artifact loading (`dnd_rf_model.pkl`, `dnd_vectorizer.pkl`)
- `score_message(...)` — rebuilds the 1×736 TF-IDF + metadata row and returns `predict_proba` for class 1
- `notification_action(prob)` — policy: `>= 0.65` BYPASS_DND, `0.40–0.65` SILENT_SUMMARY, `< 0.40` SUPPRESS
- `process_notification(...)` — scores → decides → routes to an action
- Action routers: `send_notification`, `add_to_summary`, `suppress_notification`

Model paths are resolved **relative to the file** (`os.path` from `__file__`), so
the module runs from any working directory, not just `app/`.

### 2. Environment — `.venv/` + `requirements.txt`
The original notebook environment was gone and the system Python lacked the ML
stack. Created a project venv pinned to **scikit-learn 1.6.1** (the version that
produced the pickles, confirmed via `_sklearn_version` in the model file) to
guarantee clean unpickling.

```
scikit-learn==1.6.1
scipy==1.17.1
joblib==1.5.3
```

### 3. Test suite — `tests/` + `requirements-dev.txt`
31 pytest tests, all passing:
- **`test_notebook_simulation_matches`** (11 cases) — regression check: replays the
  notebook's exact 11-message simulation and asserts each probability matches the
  original table to 3 decimals **and** the decision matches. Proves `predictor.py`
  is a faithful copy of the notebook logic.
- **`test_notification_action_thresholds`** (7 cases) — verifies the 0.65 / 0.40
  boundaries are inclusive on the correct side.
- **`test_process_notification_schema`** — output keys, types, value ranges.
- **`test_action_matches_decision`** (11 cases) — routed action always matches the decision.
- **`test_score_message_returns_probability`** — probability in [0, 1].

Run: `.\.venv\Scripts\python.exe -m pytest tests/ -v`

### 4. FastAPI service — `app/main.py`
`POST /predict` accepts the message + 4 metadata fields and returns
`{ message, probability, decision, action_result }`. Auto-generated interactive
docs at `/docs`.

Run: `cd app; ..\.venv\Scripts\python.exe -m uvicorn main:app --reload`

Verified end-to-end via `/docs` and live HTTP calls — e.g.
`"Dad is in the hospital"` (hour 23) → p = 0.786 → **BYPASS_DND**.

### 5. Architecture diagram — `images/architecture.png`
Portfolio diagram of the full pipeline (incoming notification → message + metadata
→ feature construction → Random Forest → urgency probability → decision policy →
`/predict`). Regenerable via `images/make_architecture.py` (matplotlib).

## Known follow-up
`/predict` takes a raw `dict`, so a missing field raises a `KeyError` and returns
**500** rather than a clean **422** validation error. Swap the payload for a
Pydantic `BaseModel` during README/polish to get typed fields in `/docs` and
automatic validation.

## Status
First deployable version of the system — the biggest remaining technical
milestone of Week 6. Next: README, results section, future improvements, GitHub polish.
