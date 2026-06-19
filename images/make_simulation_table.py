"""Render the notification simulation as a table image (images/simulation_output.png).

Runs the 11 sample messages through the real predictor and draws the
message / probability / decision table, color-coded by decision.

Run from the project root:
    .\\.venv\\Scripts\\python.exe images/make_simulation_table.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make app/predictor.py importable
APP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app")
sys.path.insert(0, APP_DIR)
from predictor import process_notification  # noqa: E402

# Same 11 messages used in DND_Bypass_NotificationSimulation.ipynb
TEST_MESSAGES = [
    {"message": "Please call me immediately",             "sender_importance": 5, "is_group_chat": 0, "hour_of_day": 23, "average_messages_per_day": 1},
    {"message": "Dad is in the hospital",                 "sender_importance": 5, "is_group_chat": 0, "hour_of_day": 1,  "average_messages_per_day": 1},
    {"message": "Can you review this tomorrow?",          "sender_importance": 3, "is_group_chat": 0, "hour_of_day": 14, "average_messages_per_day": 2},
    {"message": "Team lunch at 12?",                      "sender_importance": 2, "is_group_chat": 1, "hour_of_day": 10, "average_messages_per_day": 15},
    {"message": "LOL that's hilarious",                   "sender_importance": 2, "is_group_chat": 1, "hour_of_day": 20, "average_messages_per_day": 40},
    {"message": "The server is down, please check ASAP",  "sender_importance": 5, "is_group_chat": 0, "hour_of_day": 2,  "average_messages_per_day": 1},
    {"message": "Don't forget to bring snacks tomorrow",  "sender_importance": 2, "is_group_chat": 1, "hour_of_day": 18, "average_messages_per_day": 10},
    {"message": "Emergency at home, call me now",         "sender_importance": 5, "is_group_chat": 0, "hour_of_day": 3,  "average_messages_per_day": 1},
    {"message": "Meeting moved to 3 PM",                  "sender_importance": 3, "is_group_chat": 0, "hour_of_day": 9,  "average_messages_per_day": 3},
    {"message": "Are you free this weekend?",             "sender_importance": 2, "is_group_chat": 0, "hour_of_day": 17, "average_messages_per_day": 2},
    {"message": "Can you call me when you get a chance?", "sender_importance": 5, "is_group_chat": 0, "hour_of_day": 22, "average_messages_per_day": 2},
]

DECISION_COLOR = {
    "BYPASS_DND": "#D98C8C",
    "SILENT_SUMMARY": "#E4C97A",
    "SUPPRESS": "#A9B7C5",
}

rows = []
for msg in TEST_MESSAGES:
    out = process_notification(**msg)
    rows.append((out["message"], out["probability"], out["decision"]))

rows.sort(key=lambda r: r[1], reverse=True)

# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.axis("off")
ax.set_title("Notification Simulation — Model Output", fontsize=15,
             fontweight="bold", color="#1A2330", pad=16)

headers = ["Message", "Probability", "Decision"]
cell_text = [[m, f"{p:.3f}", d] for (m, p, d) in rows]

table = ax.table(
    cellText=cell_text,
    colLabels=headers,
    colWidths=[0.6, 0.16, 0.24],
    cellLoc="left",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1, 1.6)

# Header styling
for c in range(len(headers)):
    cell = table[0, c]
    cell.set_facecolor("#4A6FA5")
    cell.set_text_props(color="white", fontweight="bold")

# Body styling: color the decision cell, center prob + decision
for i, (_m, _p, d) in enumerate(rows, start=1):
    table[i, 1].set_text_props(ha="center")
    table[i, 2].set_facecolor(DECISION_COLOR[d])
    table[i, 2].set_text_props(ha="center", fontweight="bold", color="#1A1A1A")
    if i % 2 == 0:
        table[i, 0].set_facecolor("#F2F4F8")
        table[i, 1].set_facecolor("#F2F4F8")

out_path = os.path.join(os.path.dirname(__file__), "simulation_output.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out_path}")
