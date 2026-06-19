import re
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
INPUT  = DATA / "raw" / "augmented_chat.txt"
OUTPUT = DATA / "chats.csv"

MSG_PATTERN = re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}:\d{2}[  â€¯]+[AP]M)\] ([^:]+): (.*)")

# Mojibake: UTF-8 bytes of U+202F (narrow no-break space) misread as cp1252 → â€¯
NARROW_NBSP_MOJIBAKE = "â€¯"

def normalize(line: str) -> str:
    line = line.rstrip("\n").rstrip("\r")
    line = line.replace(NARROW_NBSP_MOJIBAKE, " ")
    bracket = line.find("[")
    if 0 < bracket < 8:
        line = line[bracket:]
    return line

records = []
current = None

with open(INPUT, encoding="utf-8") as f:
    for line in f:
        line = normalize(line)
        m = MSG_PATTERN.match(line)
        if m:
            if current:
                records.append(current)
            date_str, time_str, sender, message = m.groups()
            current = {
                "date":      date_str,
                "time":      time_str,
                "timestamp": datetime.strptime(f"{date_str} {time_str.strip()}", "%m/%d/%y %I:%M:%S %p"),
                "sender":    sender,
                "message":   message,
            }
        elif current:
            current["message"] += " " + line.strip()

if current:
    records.append(current)

df = pd.DataFrame(records)
df = df[["timestamp", "date", "time", "sender", "message"]]
df.to_csv(OUTPUT, index=False, encoding="utf-8")

print(f"Rows      : {len(df):,}")
print(f"Columns   : {list(df.columns)}")
print(f"Senders   : {df['sender'].nunique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Saved to  : {OUTPUT}")
print()
print(df.head(5)[["timestamp","sender","message"]].to_string(index=False, max_colwidth=60))
