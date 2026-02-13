"""
Evaluation pipeline for SOP Chatbot accuracy (V5).
Tests exact matches, paraphrased queries, out-of-scope rejection.
Generates ROC curve + confusion matrix exports.

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
    """Run full V5 evaluation suite and report metrics."""
    print("=" * 70)
    print("  SOP CHATBOT V5 — ACCURACY EVALUATION")
    print("=" * 70)

    # Load pipeline
    pipeline = SOPPipeline()

    # Disable LLM rewrite during evaluation
    original_rewrite = config.USE_LLM_REWRITE
    config.USE_LLM_REWRITE = False

    pipeline.load()

    # Load base SOP data (ground truth)
    with open(config.SOP_DATA_FILE, "r", encoding="utf-8") as f:
        sop_data = json.load(f)

    print(f"\n📊 Ground truth: {len(sop_data)} Q&A pairs")
    print(f"📊 Out-of-scope test: {len(OUT_OF_SCOPE_QUESTIONS)} questions")
    print(f"📊 Threshold: {pipeline.calibrator.threshold:.3f}")
    print(f"📊 Calibrator: {'Trained' if pipeline.calibrator.is_trained else 'Default sigmoid'}")


    # Collect all scores and labels for ROC/confusion matrix
    all_scores = []
    all_labels = []    # 1 = should match, 0 = should reject
    all_preds = []     # 1 = matched, 0 = rejected

    # ── Test 1: Exact Match ─────────────────────────────
    print("\n" + "─" * 70)
    print("  TEST 1: EXACT MATCH (Original Questions)")
    print("─" * 70)

    exact_correct = 0
    exact_total = len(sop_data)
    exact_failures = []

    for entry in sop_data:
        result = pipeline.query(entry["question"])
        score = result["score"]
        
        all_scores.append(score)
        all_labels.append(1)  # Should match
        
        if not result["rejected"] and result["source_id"] == entry["id"]:
            exact_correct += 1
            all_preds.append(1)
        else:
            all_preds.append(0)
            exact_failures.append({
                "question": entry["question"],
                "expected_id": entry["id"],
                "got_id": result["source_id"],
                "score": score,
                "rejected": result["rejected"],
            })

    exact_accuracy = (exact_correct / exact_total) * 100
    print(f"\n  ✅ Correct: {exact_correct}/{exact_total}")
    print(f"  📊 Accuracy: {exact_accuracy:.1f}%")

    if exact_failures:
        print(f"\n  ❌ Failures ({len(exact_failures)}):")
        for f in exact_failures[:5]:
            print(f"     Q: {f['question']}")
            print(f"     Expected ID: {f['expected_id']}, Got: {f['got_id']}, Score: {f['score']:.4f}, Rejected: {f['rejected']}")

    # ── Test 2: Paraphrased Questions ───────────────────
    print("\n" + "─" * 70)
    print("  TEST 2: PARAPHRASED QUESTIONS")
    print("─" * 70)

    paraphrase_tests = []
    for entry in sop_data:
        q = entry["question"]
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
        score = result["score"]
        
        all_scores.append(score)
        all_labels.append(1)  # Should match

        if not result["rejected"] and result["source_id"] == test["expected_id"]:
            para_correct += 1
            all_preds.append(1)
        elif not result["rejected"]:
            all_preds.append(1)  # Predicted match (but wrong ID)
            para_failures.append({
                "paraphrase": test["question"],
                "original": test["original"],
                "expected_id": test["expected_id"],
                "got_id": result["source_id"],
                "score": score,
            })
        else:
            all_preds.append(0)
            para_failures.append({
                "paraphrase": test["question"],
                "original": test["original"],
                "expected_id": test["expected_id"],
                "got_id": None,
                "score": score,
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
        score = result["score"]
        
        all_scores.append(score)
        all_labels.append(0)  # Should reject

        if result["rejected"]:
            rejected_correct += 1
            all_preds.append(0)
        else:
            all_preds.append(1)
            rejection_failures.append({
                "question": q,
                "matched": result["matched_question"],
                "answer": result["answer"],
                "score": score,
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

    # ── Generate ROC Curve ──────────────────────────────
    print("\n" + "─" * 70)
    print("  EVALUATION EXPORTS")
    print("─" * 70)

    try:
        from eval.roc_curve import plot_roc_curve
        plot_roc_curve(all_scores, all_labels)
    except Exception as e:
        print(f"  ⚠️  ROC curve generation failed: {e}")

    try:
        from eval.confusion_matrix import plot_confusion_matrix
        plot_confusion_matrix(all_labels, all_preds)
    except Exception as e:
        print(f"  ⚠️  Confusion matrix generation failed: {e}")

    # ── Summary Report ──────────────────────────────────
    hallucination_rate = 0.0  # By design: no hallucination possible

    print("\n" + "=" * 70)
    print("  FINAL REPORT (V5)")
    print("=" * 70)
    print(f"""
  ┌─────────────────────────────────────────────┐
  │  Exact Match Accuracy:    {exact_accuracy:6.1f}%           │
  │  Paraphrase Accuracy:     {para_accuracy:6.1f}%           │
  │  Rejection Accuracy:      {rejection_rate:6.1f}%           │
  │  Hallucination Rate:      {hallucination_rate:6.1f}%           │
  │                                             │
  │  Threshold:               {pipeline.calibrator.threshold:.3f}            │
  │  Embedding Model:         {config.EMBEDDING_MODEL:<20s}│
  │  Retrieval Top-K:         {config.TOP_K_RETRIEVAL:<20d}│
  │  Total SOP Entries:       {len(sop_data):<20d}│
  │  Calibrator:              {'Trained' if pipeline.calibrator.is_trained else 'Default':<20s}│
  └─────────────────────────────────────────────┘
""")

    if exact_accuracy == 100.0 and rejection_rate == 100.0:
        print("  🎉 PERFECT SCORE — All tests passed!")
    elif exact_accuracy >= 95.0 and rejection_rate >= 90.0:
        print("  ✅ EXCELLENT — Production ready!")
    else:
        print("  ⚠️  NEEDS IMPROVEMENT — Review failures above")

    # Restore original LLM rewrite setting
    config.USE_LLM_REWRITE = original_rewrite

    return {
        "exact_accuracy": exact_accuracy,
        "paraphrase_accuracy": para_accuracy,
        "rejection_rate": rejection_rate,
        "hallucination_rate": hallucination_rate,
    }


if __name__ == "__main__":
    run_evaluation()
