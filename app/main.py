from fastapi import FastAPI

# Support both launch styles:
#   * from repo root:  uvicorn app.main:app --reload
#   * from app/ dir:   uvicorn main:app --reload
try:
    from app.predictor import process_notification
except ModuleNotFoundError:
    from predictor import process_notification

app = FastAPI(title="DND Bypass API")


@app.post("/predict")
def predict(payload: dict):

    result = process_notification(
        message=payload["message"],
        sender_importance=payload["sender_importance"],
        is_group_chat=payload["is_group_chat"],
        hour_of_day=payload["hour_of_day"],
        average_messages_per_day=payload["average_messages_per_day"]
    )

    return result
