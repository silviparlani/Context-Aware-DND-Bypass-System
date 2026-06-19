"""Generate the DND Bypass architecture diagram (images/architecture.png).

Run from anywhere:
    ../.venv/Scripts/python.exe images/make_architecture.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
STAGE = "#E8EEF7"      # neutral pipeline stages
STAGE_EDGE = "#4A6FA5"
MODEL = "#DCE9DC"      # the model
MODEL_EDGE = "#3E7C3E"
SERVE = "#F3E4D0"      # serving layer
SERVE_EDGE = "#B5803A"
BYPASS = "#D98C8C"     # decision outcomes
SUMMARY = "#E4C97A"
SUPPRESS = "#A9B7C5"
TEXT = "#1A2330"

fig, ax = plt.subplots(figsize=(8.5, 12.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.axis("off")


def box(cx, cy, w, h, text, face, edge, fontsize=12, weight="normal", color=TEXT):
    """Draw a rounded box centered at (cx, cy); return (bottom_y, top_y)."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6, facecolor=face, edgecolor=edge, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=color, zorder=3)
    return cy - h / 2, cy + h / 2


def arrow(y_from, y_to, cx=5.0):
    """Vertical arrow from y_from down to y_to."""
    ax.add_patch(FancyArrowPatch(
        (cx, y_from), (cx, y_to),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=1.8, color="#5A6577", zorder=1,
    ))


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(5, 14.5, "DND Bypass — System Architecture",
        ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)

W = 5.2
# ---------------------------------------------------------------------------
# Vertical pipeline
# ---------------------------------------------------------------------------
b1, t1 = box(5, 13.4, W, 0.85, "Incoming Notification", STAGE, STAGE_EDGE, weight="bold")
b2, t2 = box(5, 12.0, W, 0.95, "Message Text  +  Metadata\n(sender importance, group, hour, frequency)",
             STAGE, STAGE_EDGE, fontsize=10.5)
b3, t3 = box(5, 10.5, W, 0.95, "Feature Construction\n(TF-IDF  +  Context Features)", STAGE, STAGE_EDGE)
b4, t4 = box(5, 9.0, W, 0.85, "Random Forest Classifier", MODEL, MODEL_EDGE, weight="bold")
b5, t5 = box(5, 7.6, W, 0.85, "Urgency Probability  (0.0 – 1.0)", STAGE, STAGE_EDGE)

arrow(b1, t2)
arrow(b2, t3)
arrow(b3, t4)
arrow(b4, t5)

# ---------------------------------------------------------------------------
# Decision policy block
# ---------------------------------------------------------------------------
policy_top = 6.3
policy_bottom = 3.5
ax.add_patch(FancyBboxPatch(
    (5 - 4.8 / 2, policy_bottom), 4.8, policy_top - policy_bottom,
    boxstyle="round,pad=0.02,rounding_size=0.12",
    linewidth=1.6, facecolor="#F7F7FA", edgecolor="#5A6577", zorder=2,
))
ax.text(5, policy_top - 0.32, "Decision Policy", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=TEXT, zorder=3)
arrow(b5, policy_top)

# three rules: condition  ->  outcome pill
rules = [
    (">= 0.65", "BYPASS_DND", BYPASS),
    ("0.40 – 0.65", "SILENT_SUMMARY", SUMMARY),
    ("< 0.40", "SUPPRESS", SUPPRESS),
]
row_y = [5.45, 4.65, 3.9]
for (cond, outcome, col), ry in zip(rules, row_y):
    ax.text(3.15, ry, cond, ha="left", va="center", fontsize=11, color=TEXT, zorder=3)
    ax.annotate("", xy=(5.05, ry), xytext=(4.45, ry),
                arrowprops=dict(arrowstyle="-|>", color="#5A6577", lw=1.4), zorder=3)
    pill = FancyBboxPatch((5.1, ry - 0.28), 2.25, 0.56,
                          boxstyle="round,pad=0.02,rounding_size=0.28",
                          linewidth=1.2, facecolor=col, edgecolor="#3A3A3A", zorder=3)
    ax.add_patch(pill)
    ax.text(6.225, ry, outcome, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#1A1A1A", zorder=4)

# ---------------------------------------------------------------------------
# Serving layer
# ---------------------------------------------------------------------------
b7, t7 = box(5, 2.4, W, 0.95, "FastAPI Endpoint\nPOST  /predict", SERVE, SERVE_EDGE, weight="bold")
arrow(policy_bottom, t7)
ax.text(5, 1.45, "returns  { probability, decision, action_result }",
        ha="center", va="center", fontsize=9.5, color="#5A6577", style="italic")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = os.path.join(os.path.dirname(__file__), "architecture.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
