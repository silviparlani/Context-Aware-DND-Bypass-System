"""DND Bypass — production predictor.

Bridges the notebook prototype (DND_Bypass_NotificationSimulation.ipynb) into a
reusable module that FastAPI can import. Loads the trained model artifacts,
scores an incoming message, maps the probability to a decision, and routes it to
the matching action.
"""

import os

import joblib
from scipy.sparse import csr_matrix, hstack

# ----------------------------------------------------------------------------
# Artifact loading
# ----------------------------------------------------------------------------
# Resolve the models directory relative to this file so the module works no
# matter what the current working directory is.
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

model = joblib.load(os.path.join(MODELS_DIR, "dnd_rf_model.pkl"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "dnd_vectorizer.pkl"))


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def score_message(
    message,
    sender_importance,
    is_group_chat,
    hour_of_day,
    average_messages_per_day
):

    X_text = vectorizer.transform([message])

    X_meta = csr_matrix([[
        sender_importance,
        is_group_chat,
        hour_of_day,
        average_messages_per_day
    ]])

    X_combined = hstack([X_text, X_meta])

    probability = model.predict_proba(X_combined)[0][1]

    return probability


# ----------------------------------------------------------------------------
# Decision policy
# ----------------------------------------------------------------------------
def notification_action(prob):

    if prob >= 0.65:
        return "BYPASS_DND"

    elif prob >= 0.40:
        return "SILENT_SUMMARY"

    else:
        return "SUPPRESS"


# ----------------------------------------------------------------------------
# Actions (placeholders — replaced by real integrations later)
# ----------------------------------------------------------------------------
def send_notification(message):
    return {
        "action": "BYPASS_DND",
        "result": f"Notification delivered: {message}"
    }


def add_to_summary(message):
    return {
        "action": "SILENT_SUMMARY",
        "result": f"Added to summary: {message}"
    }


def suppress_notification(message):
    return {
        "action": "SUPPRESS",
        "result": "Notification suppressed"
    }


# ----------------------------------------------------------------------------
# End-to-end processing
# ----------------------------------------------------------------------------
def process_notification(
    message,
    sender_importance,
    is_group_chat,
    hour_of_day,
    average_messages_per_day
):

    probability = score_message(
        message,
        sender_importance,
        is_group_chat,
        hour_of_day,
        average_messages_per_day
    )

    decision = notification_action(probability)

    if decision == "BYPASS_DND":
        action_result = send_notification(message)

    elif decision == "SILENT_SUMMARY":
        action_result = add_to_summary(message)

    else:
        action_result = suppress_notification(message)

    return {
        "message": message,
        "probability": round(float(probability), 3),
        "decision": decision,
        "action_result": action_result
    }


# ----------------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------------
if __name__ == "__main__":

    result = process_notification(
        message="Dad is in the hospital",
        sender_importance=5,
        is_group_chat=0,
        hour_of_day=23,
        average_messages_per_day=1
    )

    print(result)
