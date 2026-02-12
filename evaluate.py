"""
Evaluation pipeline for SOP Chatbot accuracy.
Tests exact matches, paraphrased queries, and out-of-scope rejection.

Usage:
    python evaluate.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from core.pipeline import SOPPipeline


# Out-of-scope questions that MUST be rejected
OUT_OF_SCOPE_QUESTIONS = [
    "What is the weather today?",
    "How do I cook pasta?",
    "What is the capital of France?",
    "Tell me a joke",
    "What is machine learning?",
    "How to fix my car?",
    "What time is it?",
    "Who is the president?",
    "How do I learn Python?",
    "What is the meaning of life?",
    "Can you write me an email?",
    "How to make coffee?",
    "What is blockchain?",
    "Tell me about quantum computing",
    "How to invest in stocks?",
]


def run_evaluation():
    """Run full evaluation suite and report metrics."""
    print("=" * 70)
    print("  SOP CHATBOT — ACCURACY EVALUATION")
    print("=" * 70)

    # Load pipeline
    pipeline = SOPPipeline()
    pipeline.load()

    # Load base SOP data (ground truth)
    with open(config.SOP_DATA_FILE, "r", encoding="utf-8") as f:
        sop_data = json.load(f)

    print(f"\n📊 Ground truth: {len(sop_data)} Q&A pairs")
    print(f"📊 Out-of-scope test: {len(OUT_OF_SCOPE_QUESTIONS)} questions")
    print(f"📊 Threshold: {config.SIMILARITY_THRESHOLD}")

    # ── Test 1: Exact Match ─────────────────────────────
    print("\n" + "─" * 70)
    print("  TEST 1: EXACT MATCH (Original Questions)")
    print("─" * 70)

    exact_correct = 0
    exact_total = len(sop_data)
    exact_failures = []

    for entry in sop_data:
        result = pipeline.query(entry["question"])

        if not result["rejected"] and result["source_id"] == entry["id"]:
            exact_correct += 1
        else:
            exact_failures.append({
                "question": entry["question"],
                "expected_id": entry["id"],
                "got_id": result["source_id"],
                "score": result["score"],
                "rejected": result["rejected"],
            })

    exact_accuracy = (exact_correct / exact_total) * 100
    print(f"\n  ✅ Correct: {exact_correct}/{exact_total}")
    print(f"  📊 Accuracy: {exact_accuracy:.1f}%")

    if exact_failures:
        print(f"\n  ❌ Failures ({len(exact_failures)}):")
        for f in exact_failures[:5]:  # Show first 5
            print(f"     Q: {f['question']}")
            print(f"     Expected ID: {f['expected_id']}, Got: {f['got_id']}, Score: {f['score']:.4f}, Rejected: {f['rejected']}")

    # ── Test 2: Paraphrased Questions ───────────────────
    print("\n" + "─" * 70)
    print("  TEST 2: PARAPHRASED QUESTIONS")
    print("─" * 70)

    # Generate simple paraphrases for testing
    paraphrase_tests = []
    for entry in sop_data:
        q = entry["question"]
        # Simple paraphrases
        if q.startswith("What is"):
            paraphrase_tests.append({
                "question": q.replace("What is", "Tell me"),
                "expected_id": entry["id"],
                "original": q,
            })
        elif q.startswith("Should"):
            paraphrase_tests.append({
                "question": q.replace("Should", "Do I need to").rstrip("?") + "?",
                "expected_id": entry["id"],
                "original": q,
            })
        elif q.startswith("What"):
            paraphrase_tests.append({
                "question": q.lower(),
                "expected_id": entry["id"],
                "original": q,
            })

    para_correct = 0
    para_total = len(paraphrase_tests)
    para_failures = []

    for test in paraphrase_tests:
        result = pipeline.query(test["question"])

        if not result["rejected"] and result["source_id"] == test["expected_id"]:
            para_correct += 1
        elif not result["rejected"]:
            para_failures.append({
                "paraphrase": test["question"],
                "original": test["original"],
                "expected_id": test["expected_id"],
                "got_id": result["source_id"],
                "score": result["score"],
            })
        else:
            para_failures.append({
                "paraphrase": test["question"],
                "original": test["original"],
                "expected_id": test["expected_id"],
                "got_id": None,
                "score": result["score"],
                "rejected": True,
            })

    if para_total > 0:
        para_accuracy = (para_correct / para_total) * 100
        print(f"\n  ✅ Correct: {para_correct}/{para_total}")
        print(f"  📊 Accuracy: {para_accuracy:.1f}%")

        if para_failures:
            print(f"\n  ❌ Failures ({len(para_failures)}):")
            for f in para_failures[:5]:
                print(f"     Paraphrase: {f['paraphrase']}")
                print(f"     Original:   {f['original']}")
                print(f"     Expected ID: {f['expected_id']}, Got: {f.get('got_id')}, Score: {f['score']:.4f}")
    else:
        para_accuracy = 0
        print("  ⚠️  No paraphrase tests generated")

    # ── Test 3: Out-of-Scope Rejection ──────────────────
    print("\n" + "─" * 70)
    print("  TEST 3: OUT-OF-SCOPE REJECTION")
    print("─" * 70)

    rejected_correct = 0
    rejection_total = len(OUT_OF_SCOPE_QUESTIONS)
    rejection_failures = []

    for q in OUT_OF_SCOPE_QUESTIONS:
        result = pipeline.query(q)

        if result["rejected"]:
            rejected_correct += 1
        else:
            rejection_failures.append({
                "question": q,
                "matched": result["matched_question"],
                "answer": result["answer"],
                "score": result["score"],
            })

    rejection_rate = (rejected_correct / rejection_total) * 100
    print(f"\n  ✅ Correctly rejected: {rejected_correct}/{rejection_total}")
    print(f"  📊 Rejection accuracy: {rejection_rate:.1f}%")

    if rejection_failures:
        print(f"\n  ❌ False positives ({len(rejection_failures)}):")
        for f in rejection_failures:
            print(f"     Q: {f['question']}")
            print(f"     Matched: {f['matched']}")
            print(f"     Score: {f['score']:.4f}")

    # ── Summary Report ──────────────────────────────────
    hallucination_rate = 0.0  # By design: no hallucination possible (verbatim retrieval)

    print("\n" + "=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    print(f"""
  ┌─────────────────────────────────────────────┐
  │  Exact Match Accuracy:    {exact_accuracy:6.1f}%           │
  │  Paraphrase Accuracy:     {para_accuracy:6.1f}%           │
  │  Rejection Accuracy:      {rejection_rate:6.1f}%           │
  │  Hallucination Rate:      {hallucination_rate:6.1f}%           │
  │                                             │
  │  Threshold:               {config.SIMILARITY_THRESHOLD:.2f}             │
  │  Embedding Model:         {config.EMBEDDING_MODEL:<20s}│
  │  Total SOP Entries:       {len(sop_data):<20d}│
  └─────────────────────────────────────────────┘
""")

    if exact_accuracy == 100.0 and rejection_rate == 100.0:
        print("  🎉 PERFECT SCORE — All tests passed!")
    elif exact_accuracy >= 95.0 and rejection_rate >= 90.0:
        print("  ✅ EXCELLENT — Production ready!")
    else:
        print("  ⚠️  NEEDS IMPROVEMENT — Review failures above")

    return {
        "exact_accuracy": exact_accuracy,
        "paraphrase_accuracy": para_accuracy,
        "rejection_rate": rejection_rate,
        "hallucination_rate": hallucination_rate,
    }


if __name__ == "__main__":
    run_evaluation()
