from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
CLEAN  = DATA / "chats_clean.csv"
SAMPLE = DATA / "chats_sample.csv"

SYNTHETIC_MESSAGES = {
    "Call me ASAP", "Please call me back right now", "Pick up your phone",
    "Why aren't you answering??", "I've been calling you for an hour",
    "Please answer my calls", "Call me the second you see this",
    "Can you please just call me", "I need help", "I need help right now",
    "Please I need help", "HELP", "SOS", "I'm in trouble", "I'm not okay",
    "I'm not okay right now", "I'm scared", "I'm freaking out please help",
    "I'm hurt, need help", "I can't breathe properly, help",
    "I think I need to go to the hospital", "This is an emergency",
    "This is an emergency please respond", "Emergency please call me",
    "URGENT", "URGENT please call", "This can't wait, call me now",
    "This is not a joke, call me", "Please don't ignore this",
    "Something happened, call me", "Something's wrong, please call",
    "Something serious happened", "Had an accident, please call",
    "I think I'm having a panic attack", "I don't feel safe right now",
    "Not safe right now", "I'm outside and I need help", "I'm stuck, please help",
    "Are you okay? Please respond", "Are you safe?", "Please tell me you're okay",
    "Where are you?? I need you NOW", "Please check on me", "Please come ASAP",
    "I need you here right now", "I need you right now please", "Need you urgently",
    "Can you talk? It's urgent", "Please please call me",
    "Please wake up it's urgent", "I need to talk to you right now",
    "Can someone please help me", "Please respond ASAP",
    "Please pick up, it's important",
}

for path in [CLEAN, SAMPLE]:
    df = pd.read_csv(path)
    df["source"] = df["message"].apply(
        lambda m: "synthetic" if str(m).strip() in SYNTHETIC_MESSAGES else "real"
    )
    df.to_csv(path, index=False, encoding="utf-8")

    counts = df["source"].value_counts()
    print(f"{path.split(chr(92))[-1]}")
    print(f"  real      : {counts.get('real', 0):,}")
    print(f"  synthetic : {counts.get('synthetic', 0):,}")
    print()
