"""
Append synthetic group-chat senders (group_1..group_5) to chats_clean.csv and
chats_sample_features.csv. Each group has a tone profile matching its priority
weight on the 1-5 scale used for Person_N senders.

  group_1 -> 5  family-emergency group (mostly urgent)
  group_2 -> 4  work / team-lead channel (important coordination + occasional urgency)
  group_3 -> 3  close-friends group (mixed, support-leaning)
  group_4 -> 2  society/neighbours group (mostly casual logistics)
  group_5 -> 1  random/social/memes group (pure casual)

chats_sample_baseline.csv is intentionally NOT touched -- baseline metrics stay frozen.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
CLEAN = DATA / "chats_clean.csv"
FEATURES = DATA / "chats_sample_features.csv"

random.seed(42)

GROUP_PROFILES = {
    "group_1": {
        "priority": 5,
        "label_weights": {1: 0.7, 0: 0.3},
        "messages_urgent": [
            "Nani is in the hospital, please come",
            "Dad just had a fall, we're rushing to the ER",
            "Mom's BP shot up, doctor is on the way",
            "Anyone reachable? bua is unwell, need someone now",
            "Please call home, it's urgent",
            "Hospital admit ho gaye papa, please come fast",
            "We need everyone at the hospital ASAP",
            "Family emergency, please respond",
            "Dadi is being shifted to ICU, pray",
            "Bhaiya had an accident, on the way to AIIMS",
            "Please pick up, it's about mom",
            "Need someone to come to the hospital right now",
        ],
        "messages_normal": [
            "Reached safely",
            "She's stable now, thank god",
            "Doctor said she'll be discharged tomorrow",
            "Will share updates here",
            "Thank you everyone for praying",
        ],
    },
    "group_2": {
        "priority": 4,
        "label_weights": {1: 0.3, 0: 0.7},
        "messages_urgent": [
            "Production is down, all hands please",
            "Client escalation, jump on the bridge call now",
            "Pager just fired, anyone on it?",
            "Sev1 raised, war room link incoming",
            "Need a fix in the next hour or we miss the SLA",
            "@channel critical bug in prod, please respond",
            "Boss is calling an emergency standup in 10 mins",
        ],
        "messages_normal": [
            "Standup at 10, please join on time",
            "Sharing the deck for tomorrow's review",
            "Sprint planning moved to Thursday",
            "PRs to review before EOD please",
            "Lunch order going out, drop names",
            "Reminder: TPS reports due today",
            "Anyone has the meeting notes from yesterday",
            "Coffee break in the pantry, come join",
            "Long weekend coming up, plan handoffs",
            "Demo recording uploaded to the shared drive",
            "Quick check, server restart at 6pm",
            "Reminder to file your timesheets",
            "All-hands moved to 4pm",
        ],
    },
    "group_3": {
        "priority": 3,
        "label_weights": {1: 0.15, 0: 0.85},
        "messages_urgent": [
            "Guys I'm not okay, can someone call",
            "Really need to talk, anyone free?",
            "Having a rough day, please someone reply",
            "I'm spiralling, please help",
        ],
        "messages_normal": [
            "Brunch on Saturday? Confirm in 12 hours",
            "Anyone watching the match tonight",
            "Movie at 9, who's in",
            "Birthday plan for Sneha, ideas please",
            "Throwback to last year's trip <3",
            "Sending so much love today",
            "Anniversary gift suggestions?",
            "Who's coming to the housewarming",
            "Group call tonight at 10?",
            "Sharing the trip itinerary, check and confirm",
            "Cake order placed for Saturday",
            "Driver booked for the trip",
            "Photos from yesterday in the drive",
            "Missing you all so much",
        ],
    },
    "group_4": {
        "priority": 2,
        "label_weights": {1: 0.05, 0: 0.95},
        "messages_urgent": [
            "Water tank is overflowing in B-wing, urgent",
            "Lift stuck on 7th floor, someone inside please respond",
        ],
        "messages_normal": [
            "Maintenance bill due by 10th",
            "RWA meeting on Sunday at 11 am",
            "Plumber coming tomorrow, please be available",
            "Power cut from 2pm to 4pm tomorrow",
            "Diwali decoration volunteers needed",
            "Society parking sticker renewal reminder",
            "New security guard joining from Monday",
            "Holi celebration this Sunday",
            "Lift servicing scheduled for 3pm",
            "Watchman changed shift today",
            "Newspaper guy is on leave for two days",
        ],
    },
    "group_5": {
        "priority": 1,
        "label_weights": {0: 1.0},
        "messages_urgent": [],
        "messages_normal": [
            "lol check this reel",
            "have you guys seen this meme",
            "haha that's hilarious",
            "this trend is everywhere",
            "weekly meme dump incoming",
            "another monday another meeting",
            "instagram is wild today",
            "this song is stuck in my head",
            "binge watched the whole series",
            "spam your wordle scores",
            "rate my coffee",
            "who's keeping up with the new season",
            "this filter is so cursed lol",
            "new episode drops tonight",
        ],
    },
}

START_DATE = datetime(2024, 6, 1)
END_DATE = datetime(2026, 4, 30)
DATE_RANGE_SECS = int((END_DATE - START_DATE).total_seconds())


def random_dt() -> datetime:
    return START_DATE + timedelta(seconds=random.randint(0, DATE_RANGE_SECS))


def pick_message(profile: dict) -> tuple[str, int]:
    labels = list(profile["label_weights"].keys())
    weights = list(profile["label_weights"].values())
    label = random.choices(labels, weights=weights, k=1)[0]
    pool = profile["messages_urgent"] if label == 1 else profile["messages_normal"]
    if not pool:
        pool = profile["messages_normal"]
        label = 0
    return random.choice(pool), label


def clean_row(dt: datetime, sender: str, msg: str) -> dict:
    return {
        "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "date": f"{dt.month}/{dt.day}/{str(dt.year)[2:]}",
        "time": dt.strftime("%I:%M:%S %p").lstrip("0"),
        "sender": sender,
        "message": msg,
        "source": "synthetic",
    }


def sample_row(dt: datetime, sender: str, msg: str, label: int) -> dict:
    return {
        "timestamp": dt.strftime("%d-%m-%Y %H:%M"),
        "date": dt.strftime("%d-%m-%Y"),
        "time": dt.strftime("%I:%M:%S %p").lstrip("0"),
        "sender": sender,
        "message": msg,
        "label": label,
        "source": "synthetic",
    }


def append_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            writer.writerow(r)


def main():
    clean_rows = []
    sample_rows = []

    for sender, profile in GROUP_PROFILES.items():
        for _ in range(10):
            msg, _ = pick_message(profile)
            clean_rows.append(clean_row(random_dt(), sender, msg))
        for _ in range(5):
            msg, label = pick_message(profile)
            sample_rows.append(sample_row(random_dt(), sender, msg, label))

    clean_fields = ["timestamp", "date", "time", "sender", "message", "source"]
    sample_fields = ["timestamp", "date", "time", "sender", "message", "label", "source"]

    append_rows(CLEAN, clean_rows, clean_fields)
    append_rows(FEATURES, sample_rows, sample_fields)

    print(f"Appended {len(clean_rows)} rows to {CLEAN.name}")
    print(f"Appended {len(sample_rows)} rows to {FEATURES.name}")
    print("\nGroup priority mapping (extends Person_N hierarchy):")
    for sender, profile in GROUP_PROFILES.items():
        print(f"  {sender}: {profile['priority']}")


if __name__ == "__main__":
    main()
