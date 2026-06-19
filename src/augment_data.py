import random
from datetime import datetime, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
INPUT  = DATA / "raw" / "combined_chat.txt"
OUTPUT = DATA / "raw" / "augmented_chat.txt"

EMERGENCY_MESSAGES = [
    # Urgent calls
    "Call me ASAP",
    "Please call me back right now",
    "Pick up your phone",
    "Why aren't you answering??",
    "I've been calling you for an hour",
    "Please answer my calls",
    "Call me the second you see this",
    "Can you please just call me",
    # Direct distress
    "I need help",
    "I need help right now",
    "Please I need help",
    "HELP",
    "SOS",
    "I'm in trouble",
    "I'm not okay",
    "I'm not okay right now",
    "I'm scared",
    "I'm freaking out please help",
    "I'm hurt, need help",
    "I can't breathe properly, help",
    "I think I need to go to the hospital",
    # Emergency flags
    "This is an emergency",
    "This is an emergency please respond",
    "Emergency please call me",
    "URGENT",
    "URGENT please call",
    "This can't wait, call me now",
    "This is not a joke, call me",
    "Please don't ignore this",
    # Situation updates
    "Something happened, call me",
    "Something's wrong, please call",
    "Something serious happened",
    "Had an accident, please call",
    "I think I'm having a panic attack",
    "I don't feel safe right now",
    "Not safe right now",
    "I'm outside and I need help",
    "I'm stuck, please help",
    # Checking in / pleading
    "Are you okay? Please respond",
    "Are you safe?",
    "Please tell me you're okay",
    "Where are you?? I need you NOW",
    "Please check on me",
    "Please come ASAP",
    "I need you here right now",
    "I need you right now please",
    "Need you urgently",
    "Can you talk? It's urgent",
    "Please please call me",
    "Please wake up it's urgent",
    "I need to talk to you right now",
    "Can someone please help me",
    "Please respond ASAP",
    "Please pick up, it's important",
]

NAMES = [
    "Silvi", "Ayushika Anand", "Nidhi Malpani", "Isha Shukla",
    "Rohit", "Mom", "Dad", "Priya", "Arjun", "Neha",
]

START_DATE = datetime(2022, 12, 25)
END_DATE   = datetime(2026, 4, 30)
DATE_RANGE_SECS = int((END_DATE - START_DATE).total_seconds())


def random_timestamp() -> str:
    dt = START_DATE + timedelta(seconds=random.randint(0, DATE_RANGE_SECS))
    month, day = dt.month, dt.day
    year  = str(dt.year)[2:]
    hour  = dt.hour % 12 or 12
    ampm  = "AM" if dt.hour < 12 else "PM"
    return f"[{month}/{day}/{year}, {hour}:{dt.strftime('%M:%S')} {ampm}]"


with open(INPUT, encoding="utf-8") as f:
    lines = f.readlines()

message_count = sum(1 for l in lines if l.startswith("["))
target = round(message_count * 0.10)

emergency_lines = [
    f"{random_timestamp()} {random.choice(NAMES)}: {random.choice(EMERGENCY_MESSAGES)}\n"
    for _ in range(target)
]

insert_positions = sorted(random.sample(range(len(lines)), target))

new_lines = list(lines)
for offset, (pos, msg) in enumerate(zip(insert_positions, emergency_lines)):
    new_lines.insert(pos + offset, msg)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

new_msg_count = sum(1 for l in new_lines if l.startswith("["))
print(f"Original messages : {message_count:,}")
print(f"Emergency inserted: {target:,}")
print(f"Total messages    : {new_msg_count:,}")
print(f"Saved to          : {OUTPUT}")
