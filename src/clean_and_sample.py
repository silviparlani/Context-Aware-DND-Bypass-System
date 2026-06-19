import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
INPUT   = DATA / "chats.csv"
CLEAN   = DATA / "chats_clean.csv"
SAMPLE  = DATA / "chats_sample.csv"

df = pd.read_csv(INPUT)
print(f"Loaded : {len(df):,} rows")

# --- Junk rules ---
# 1. Media/file placeholders
mask_media = df["message"].str.contains(r"\bomitted\b", case=False, na=False)

# 2. WhatsApp system messages (encryption notice, group creation)
mask_system = df["message"].str.contains(
    r"end-to-end encrypted|You created this group|Missed voice call",
    case=False, na=False
)

# 3. Group admin events (icon/description/subject changes)
mask_group_events = df["message"].str.contains(
    r"^[â€Ž¯]*\s*(?:You changed this group|Person_\d+ changed the group|Person_\d+ changed this group)",
    case=False, na=False
)

# 4. Deleted messages
mask_deleted = df["message"].str.contains(
    r"This message was deleted|You deleted this message",
    case=False, na=False
)

# 5. Empty / whitespace-only messages
mask_empty = df["message"].str.strip().eq("") | df["message"].isna()

junk = mask_media | mask_system | mask_group_events | mask_deleted | mask_empty

print(f"\nJunk breakdown:")
print(f"  Media placeholders   : {mask_media.sum():,}")
print(f"  System messages      : {mask_system.sum():,}")
print(f"  Group admin events   : {mask_group_events.sum():,}")
print(f"  Deleted messages     : {mask_deleted.sum():,}")
print(f"  Empty messages       : {mask_empty.sum():,}")
print(f"  Total junk           : {junk.sum():,}")

df_clean = df[~junk].reset_index(drop=True)
df_clean.to_csv(CLEAN, index=False, encoding="utf-8")
print(f"\nAfter cleaning : {len(df_clean):,} rows  ->saved to chats_clean.csv")

# --- Sample 500 ---
df_sample = df_clean.sample(n=500, random_state=42).reset_index(drop=True)
df_sample.to_csv(SAMPLE, index=False, encoding="utf-8")
print(f"Sample         : {len(df_sample):,} rows  ->saved to chats_sample.csv")
print()
print(df_sample[["timestamp", "sender", "message"]].head(10).to_string(index=False, max_colwidth=70))
