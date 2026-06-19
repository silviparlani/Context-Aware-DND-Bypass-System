"""Test suite for app/predictor.py.

The headline test (`test_notebook_simulation_matches`) is a regression check:
it feeds the exact 11 messages from DND_Bypass_NotificationSimulation.ipynb
through the extracted module and asserts the probabilities and decisions match
the table the notebook produced. This proves predictor.py is a faithful copy of
the notebook logic, not just that it runs.

The remaining tests cover the decision policy thresholds, the output schema, and
action routing in isolation.
"""

import pytest

from predictor import (
    notification_action,
    process_notification,
    score_message,
)

# ---------------------------------------------------------------------------
# Reference data: the 11 messages from the notebook, with their inputs and the
# probability / decision the notebook produced. Probabilities are the values
# from the simulation table (rounded to 3 decimals).
# ---------------------------------------------------------------------------
NOTEBOOK_CASES = [
    # (message, sender_importance, is_group_chat, hour_of_day, avg_msgs/day, prob, decision)
    ("Please call me immediately",              5, 0, 23,  1, 0.719, "BYPASS_DND"),
    ("Dad is in the hospital",                  5, 0,  1,  1, 0.846, "BYPASS_DND"),
    ("Can you review this tomorrow?",           3, 0, 14,  2, 0.133, "SUPPRESS"),
    ("Team lunch at 12?",                       2, 1, 10, 15, 0.139, "SUPPRESS"),
    ("LOL that's hilarious",                    2, 1, 20, 40, 0.073, "SUPPRESS"),
    ("The server is down, please check ASAP",   5, 0,  2,  1, 0.893, "BYPASS_DND"),
    ("Don't forget to bring snacks tomorrow",   2, 1, 18, 10, 0.060, "SUPPRESS"),
    ("Emergency at home, call me now",          5, 0,  3,  1, 0.861, "BYPASS_DND"),
    ("Meeting moved to 3 PM",                   3, 0,  9,  3, 0.176, "SUPPRESS"),
    ("Are you free this weekend?",              2, 0, 17,  2, 0.112, "SUPPRESS"),
    ("Can you call me when you get a chance?",  5, 0, 22,  2, 0.417, "SILENT_SUMMARY"),
]


# ---------------------------------------------------------------------------
# Regression: predictor.py reproduces the notebook simulation exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message, sender, group, hour, avg, expected_prob, expected_decision",
    NOTEBOOK_CASES,
    ids=[c[0] for c in NOTEBOOK_CASES],
)
def test_notebook_simulation_matches(
    message, sender, group, hour, avg, expected_prob, expected_decision
):
    result = process_notification(
        message=message,
        sender_importance=sender,
        is_group_chat=group,
        hour_of_day=hour,
        average_messages_per_day=avg,
    )

    # Probability must round to the same 3-decimal value the notebook reported.
    assert result["probability"] == pytest.approx(expected_prob, abs=1e-3), (
        f"{message!r}: got {result['probability']}, expected {expected_prob}"
    )
    assert result["decision"] == expected_decision


# ---------------------------------------------------------------------------
# Decision policy thresholds (>=0.65 BYPASS, >=0.40 SUMMARY, else SUPPRESS)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prob, expected",
    [
        (1.00, "BYPASS_DND"),
        (0.65, "BYPASS_DND"),      # lower boundary, inclusive
        (0.6499, "SILENT_SUMMARY"),
        (0.50, "SILENT_SUMMARY"),
        (0.40, "SILENT_SUMMARY"),  # lower boundary, inclusive
        (0.3999, "SUPPRESS"),
        (0.00, "SUPPRESS"),
    ],
)
def test_notification_action_thresholds(prob, expected):
    assert notification_action(prob) == expected


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
def test_process_notification_schema():
    result = process_notification(
        message="Dad is in the hospital",
        sender_importance=5,
        is_group_chat=0,
        hour_of_day=1,
        average_messages_per_day=1,
    )

    assert set(result) == {"message", "probability", "decision", "action_result"}
    assert result["message"] == "Dad is in the hospital"
    assert isinstance(result["probability"], float)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["decision"] in {"BYPASS_DND", "SILENT_SUMMARY", "SUPPRESS"}
    assert set(result["action_result"]) == {"action", "result"}


# ---------------------------------------------------------------------------
# Action routing: the action taken must match the decision
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message, sender, group, hour, avg, expected_decision",
    [(m, s, g, h, a, d) for (m, s, g, h, a, _p, d) in NOTEBOOK_CASES],
    ids=[c[0] for c in NOTEBOOK_CASES],
)
def test_action_matches_decision(
    message, sender, group, hour, avg, expected_decision
):
    result = process_notification(
        message=message,
        sender_importance=sender,
        is_group_chat=group,
        hour_of_day=hour,
        average_messages_per_day=avg,
    )
    assert result["action_result"]["action"] == result["decision"]


# ---------------------------------------------------------------------------
# score_message returns a bare probability in [0, 1]
# ---------------------------------------------------------------------------
def test_score_message_returns_probability():
    prob = score_message(
        message="The server is down, please check ASAP",
        sender_importance=5,
        is_group_chat=0,
        hour_of_day=2,
        average_messages_per_day=1,
    )
    assert 0.0 <= float(prob) <= 1.0
