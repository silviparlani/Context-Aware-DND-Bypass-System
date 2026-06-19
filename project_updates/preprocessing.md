# Preprocessing

---

## 1. Raw Data

- Exported 4 WhatsApp chats as `.txt` files.
- Sources: individual and group conversations spanning December 2022 – May 2026.
- Total: **85,195 messages** across 4 contacts/groups.

| File | Messages |
|---|---|
| `ayushika_anand_chat.txt` | 427 |
| `bhailog_chat.txt` | 12,187 |
| `forge_chat.txt` | 52,123 |
| `isha_shukla_chat.txt` | 20,459 |

---

## 2. Consolidation

- Moved all `_chat.txt` files from their export subfolders into `data/raw/`.
- Renamed files to `<contact>_chat.txt` for clarity.
- Concatenated all four files into `combined_chat.txt` using PowerShell.

---

## 3. Data Augmentation

- **Goal:** inject ~10% emergency-type messages to create positive class examples for DND-bypass classification.
- **Script:** `src/augment_data.py`
- Generated **8,520 synthetic messages** using 55 emergency templates (e.g., *"Call me ASAP"*, *"I'm not okay"*, *"SOS"*) assigned to random senders and timestamps within the existing date range.
- Messages were inserted at random positions throughout the dataset.
- Output: `data/raw/augmented_chat.txt`

---

## 4. Anonymization

- **Script:** `src/anonymize.py`
- Replaced all 14 real sender names with anonymous placeholders (`Person_1` through `Person_14`), ordered by first appearance.
- Replacement applied in-place on `augmented_chat.txt`.

---

## 5. Parsing to DataFrame

- **Script:** `src/parse_to_dataframe.py`
- Parsed `augmented_chat.txt` into structured CSV with columns: `timestamp`, `date`, `time`, `sender`, `message`.
- Handled encoding artifacts introduced by PowerShell's cp1252/UTF-8 re-encoding:
  - Mojibake narrow no-break space (`â€¯` → ` `) before AM/PM in timestamps.
  - Leading invisible Unicode characters (U+200E, BOM) before `[` in message lines.
- Multi-line messages (continuation lines) concatenated into a single row.
- Output: `data/chats.csv` — **97,566 rows**, 14 senders.

---

## 6. Junk Removal

- **Script:** `src/clean_and_sample.py`
- Removed 4,157 rows matching the following patterns:

| Category | Count |
|---|---|
| Media placeholders (`image/audio/sticker omitted`) | 3,911 |
| Deleted messages | 208 |
| System messages (encryption notice, missed calls, group creation) | 32 |
| Group admin events (icon/description changes) | 3 |
| Empty/whitespace-only messages | 3 |

- Output: `data/chats_clean.csv` — **93,409 rows**.

---

## 7. Sampling

- **Script:** `src/clean_and_sample.py`
- Drew a random 500-row sample (`random_state=42`) from the clean dataset for iterative development.
- Output: `data/chats_sample.csv` — **500 rows**.

---

## 8. Source Column

- **Script:** `src/add_source_column.py`
- Added a `source` column (`real` / `synthetic`) to both `chats_clean.csv` and `chats_sample.csv`.
- Synthetic rows identified by exact match against the 55 emergency message templates used in augmentation.

| Dataset | real | synthetic |
|---|---|---|
| `chats_clean.csv` | 85,502 | 7,907 |
| `chats_sample.csv` | 466 | 34 |

---

## 9. Text Cleaning

- **Script:** `src/clean_and_label.py`
- Stripped all non-printable-ASCII characters from the `message` column in both CSVs using `re.sub(r'[^\x20-\x7E]', '', text)`.
- This removed emojis and mojibake sequences (e.g., `â€™`, `ðŸ˜‚`) that resulted from PowerShell's cp1252/UTF-8 double-encoding of the original WhatsApp exports.
- Emoji-only messages became empty strings after cleaning.

---

## 10. Labeling

- **Script:** `src/clean_and_label.py`
- Added a `label` column to `chats_sample.csv` for DND-bypass classification:
  - `1` = emergency, should bypass Do Not Disturb
  - `0` = normal conversation
  - `-1` = non-English (Hindi/Marathi in Roman script) or content-free after cleaning (emoji-only)
- First 119 rows labeled manually; remaining 381 labeled programmatically.
- All synthetic messages labeled `1`; Hindi/Marathi Romanized messages and post-cleaning empty messages labeled `-1`; everything else labeled `0`.

| Label | Count | Meaning |
|---|---|---|
| `1` | 36 | Emergency |
| `0` | 394 | Normal |
| `-1` | 70 | Non-English or content-free |
