"""
Parse data/data.txt into structured JSON format.
Also generates paraphrase variants for improved retrieval accuracy.

Usage:
    python scripts/parse_data.py
"""
import json
import os
import re
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def parse_qa_file(filepath: str) -> list[dict]:
    """
    Parse a Q&A text file where questions start with 'Q:' and answers with 'A:'.
    Returns list of {"id": int, "question": str, "answer": str}.
    """
    entries = []
    current_id = 0

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Q:"):
            question = line[2:].strip()
            # Look for the answer on the next non-empty line
            i += 1
            while i < len(lines):
                answer_line = lines[i].strip()
                if answer_line.startswith("A:"):
                    answer = answer_line[2:].strip()
                    current_id += 1
                    entries.append({
                        "id": current_id,
                        "question": question,
                        "answer": answer,
                    })
                    break
                elif answer_line == "":
                    i += 1
                    continue
                else:
                    break
        i += 1

    return entries


def generate_paraphrases(question: str) -> list[str]:
    """
    Generate simple rule-based paraphrases of a question
    to improve retrieval robustness. No LLM needed.
    """
    paraphrases = []
    q = question.lower().strip().rstrip("?").strip()

    # Pattern: "What is X" → "Tell me about X", "X?"
    if q.startswith("what is "):
        subject = q[8:]
        paraphrases.append(f"Tell me about {subject}")
        paraphrases.append(subject)
        paraphrases.append(f"Explain {subject}")

    # Pattern: "What should be done" → "How to handle", "Steps for"
    if q.startswith("what should be done"):
        rest = q[19:].strip()
        paraphrases.append(f"How to handle {rest}")
        paraphrases.append(f"Steps for {rest}")

    # Pattern: "Should X" → "Do I need to X", "Is it required to X"
    if q.startswith("should "):
        rest = q[7:]
        paraphrases.append(f"Do I need to {rest}")
        paraphrases.append(f"Is it required to {rest}")
        paraphrases.append(f"Must I {rest}")

    # Pattern: "How do I" / "How to" → alternative phrasings
    if q.startswith("how do i "):
        rest = q[9:]
        paraphrases.append(f"Steps to {rest}")
        paraphrases.append(f"Process for {rest}")
    elif q.startswith("how to "):
        rest = q[7:]
        paraphrases.append(f"Steps to {rest}")
        paraphrases.append(f"Way to {rest}")

    # Pattern: "Which X" → "What X"
    if q.startswith("which "):
        rest = q[6:]
        paraphrases.append(f"What {rest}")

    # Pattern: "When should" → "When to"
    if q.startswith("when should "):
        rest = q[12:]
        paraphrases.append(f"When to {rest}")

    # Pattern: "Who is" → "Who's"
    if q.startswith("who is "):
        rest = q[7:]
        paraphrases.append(f"Who's {rest}")

    # Pattern: "Can X" → "Is X allowed", "Is it ok to X"
    if q.startswith("can "):
        rest = q[4:]
        paraphrases.append(f"Is {rest} allowed")
        paraphrases.append(f"Is it ok to {rest}")

    # Pattern: "Where are" → "Where to find", "Location of"
    if q.startswith("where are "):
        rest = q[10:]
        paraphrases.append(f"Where to find {rest}")
        paraphrases.append(f"Location of {rest}")

    # Pattern: "How many" → "Number of"
    if q.startswith("how many "):
        rest = q[9:]
        paraphrases.append(f"Number of {rest}")

    return paraphrases


def build_augmented_dataset(entries: list[dict]) -> list[dict]:
    """
    Build an augmented dataset where paraphrased questions
    all point to the same original answer.
    Returns list of {"id": int, "question": str, "answer": str, "source_id": int}
    """
    augmented = []
    aug_id = 0

    for entry in entries:
        # Add the original
        aug_id += 1
        augmented.append({
            "id": aug_id,
            "question": entry["question"],
            "answer": entry["answer"],
            "source_id": entry["id"],
        })

        # Add paraphrases
        paraphrases = generate_paraphrases(entry["question"])
        for para in paraphrases:
            aug_id += 1
            augmented.append({
                "id": aug_id,
                "question": para,
                "answer": entry["answer"],
                "source_id": entry["id"],
            })

    return augmented


def main():
    print("=" * 60)
    print("  SOP Data Parser & Paraphrase Augmentor")
    print("=" * 60)

    # Parse raw data
    print(f"\n📂 Reading: {config.RAW_DATA_FILE}")
    entries = parse_qa_file(config.RAW_DATA_FILE)
    print(f"✅ Parsed {len(entries)} Q&A pairs")

    # Save base dataset
    base_file = config.SOP_DATA_FILE
    with open(base_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved base dataset → {base_file}")

    # Generate augmented dataset
    augmented = build_augmented_dataset(entries)
    aug_file = os.path.join(config.DATA_DIR, "sop_data_augmented.json")
    with open(aug_file, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved augmented dataset → {aug_file}")
    print(f"   ({len(entries)} original + {len(augmented) - len(entries)} paraphrases = {len(augmented)} total)")

    # Show sample
    print(f"\n📋 Sample entries:")
    for entry in augmented[:5]:
        print(f"   [{entry['id']}] Q: {entry['question']}")
        print(f"        A: {entry['answer']}")
        print()

    print("✅ Data parsing complete!")


if __name__ == "__main__":
    main()
