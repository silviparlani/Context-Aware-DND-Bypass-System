import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
INPUT  = DATA / "raw" / "augmented_chat.txt"
OUTPUT = DATA / "raw" / "augmented_chat.txt"

with open(INPUT, encoding="utf-8") as f:
    lines = f.readlines()

# Collect unique names in order of first appearance
seen = {}
for line in lines:
    m = re.match(r'^\[.*?\] (.+?):', line)
    if m:
        name = m.group(1)
        if name not in seen:
            seen[name] = f"Person_{len(seen) + 1}"

print("Name mapping:")
for original, placeholder in seen.items():
    print(f"  {original!r:30} -> {placeholder}")

# Replace each name (longest first to avoid partial matches)
new_lines = []
for line in lines:
    for original, placeholder in sorted(seen.items(), key=lambda x: -len(x[0])):
        line = line.replace(original, placeholder)
    new_lines.append(line)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"\nAnonymized {len(seen)} names. Saved to {OUTPUT}")
