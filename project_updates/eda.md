# Exploratory Data Analysis

**Date:** 2026-05-05  
**Notebook:** `notebooks/DND_Bypass_EDA.ipynb`  
**Dataset:** `data/chats_sample.csv` (500 rows; label `-1` rows excluded from analysis)

---

## Analysis 1 — Time-of-Day Distribution by Label (Violin Plot)

- Converted `timestamp` to datetime; extracted `hour` as a numeric feature.
- Plotted distribution of message hours for label `0` and label `1`.

**Findings:**
- Urgent messages (label `1`) show a **bimodal distribution** peaking around ~3 AM and ~1 PM, indicating urgency is not confined to typical active hours.
- The late-night/early-morning window (00:00–05:00) is **disproportionately associated with urgency**, making time-of-day a strong candidate signal for DND override.
- Non-urgent messages (label `0`) cluster heavily around **~11 PM**, suggesting higher noise during evening hours.

---

## Analysis 2 — Label Proportion by Sender (Stacked Bar Chart)

- Grouped by `sender` × `label`, computed proportions per sender.
- Sorted senders by descending proportion of label `1`.

**Findings:**
- Some senders contribute **exclusively urgent messages**, suggesting they should receive higher bypass priority by default.
- Others show **zero urgent messages**, making them low-priority bypass candidates.
- The sender population segments naturally into: **low urgency**, **high urgency**, and **mixed urgency** senders.

---

## Analysis 3 — Message Frequency Heatmap (All Messages)

- Plotted a heatmap of message count per sender × hour.

**Findings:**
- Person 1, 14, and 3 dominate message volume but fall into low/mixed urgency categories.
- High-urgency senders (e.g., Persons 5, 7, 8, 9, 10) send **no more than 5 messages at any given hour**, consistent with rare-but-important communication patterns.
- There is a **negative correlation** between a sender's total message frequency and their urgency rate.

---

## Analysis 4 — Urgent Message Frequency Heatmap (Label = 1 Only)

- Filtered to label `1` messages and re-plotted the sender × hour heatmap.

**Findings:**
- High-urgency senders are sparse across both time slots and total volume, confirming they only message in emergencies.
- Mixed-urgency senders produce a disproportionate volume of non-urgent messages relative to urgent ones.

---

## Analysis 5 — Message Length Distribution by Label (Violin Plot)

- Created a `message_length` feature from character count of the `message` column.
- Plotted a violin plot of message length for label `0` and label `1`.

**Findings:**
- Non-urgent messages (label `0`) show a **wide distribution** with a peak at shorter lengths but a long tail extending to very long messages.
- Urgent messages (label `1`) show a **more concentrated distribution** — often short and to-the-point.
- Both labels overlap heavily at shorter lengths, confirming message length is a weak signal in isolation.

---

## Key Insights

### Sender & Time Signals

**Frequency vs Urgency Tradeoff:** High-frequency senders tend to produce a larger proportion of non-urgent messages. Message volume alone is not a reliable proxy for importance.

**Rare but Important Senders:** Some low-frequency senders show a higher proportion of urgent messages. Sender importance should not be based purely on interaction frequency.

**Heterogeneous Sender Behavior:** Some users generate high noise; others generate sparse but potentially critical messages. This supports building a personalized sender importance feature rather than applying a global rule.

**Combined Signals Necessary:** Urgent messages are sparse and unevenly distributed across both senders and time, reinforcing that no single feature is sufficient — the system needs to combine sender identity, time-of-day, and text signals.

### Message Length Signals

**Brevity as a Weak Signal:** Urgent messages tend to have a more concentrated length distribution, often shorter than non-urgent messages. Brevity is a weak but potentially useful supporting feature.

**Long Content is Likely Non-Urgent:** Non-urgent messages exhibit a much wider length distribution including very long messages, suggesting long-form content is more likely informational or conversational.

**Length Is Not Reliable Alone:** Both short and moderately long messages can be urgent. Message length must be combined with other features and should not be used as a primary decision signal.
