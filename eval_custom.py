"""
Custom evaluation using eval.txt test cases.
Parses Q/Expected pairs and tests them against the SOP pipeline.

Usage:
    python eval_custom.py [path_to_eval.txt]
"""
import os
import sys
import re
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from core.pipeline import SOPPipeline

# The rejection message (normalized for comparison)
REJECTION_KEYWORDS = ["not covered", "contact hr"]


def parse_eval_file(filepath: str) -> list[dict]:
    """Parse eval.txt into list of {question, expected} dicts."""
    tests = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_q = None
    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            current_q = line[2:].strip()
        elif line.startswith("Expected:") and current_q:
            expected = line[len("Expected:"):].strip()
            tests.append({"question": current_q, "expected": expected})
            current_q = None

    return tests


def is_rejection(text: str) -> bool:
    """Check if the text is a rejection/not-covered response."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in REJECTION_KEYWORDS)


def answer_matches(retrieved: str, expected: str) -> bool:
    """
    Check if the retrieved answer semantically matches the expected answer.
    Uses substring matching on key phrases since answers are short SOP entries.
    """
    # Normalize both
    retrieved_lower = retrieved.lower().strip()
    expected_lower = expected.lower().strip()

    # Both are rejections
    if is_rejection(expected) and is_rejection(retrieved):
        return True

    # One is rejection, other isn't
    if is_rejection(expected) != is_rejection(retrieved):
        return False

    # Extract key phrases from expected answer for matching
    # Remove common filler words for comparison
    expected_keywords = set(re.findall(r'\b\w{3,}\b', expected_lower))
    # Remove very common words
    stop_words = {"the", "and", "for", "are", "that", "this", "with", "from", "should", "must", "please", "yes", "not"}
    expected_keywords -= stop_words

    if not expected_keywords:
        return expected_lower in retrieved_lower

    # Check how many key phrases from expected appear in retrieved
    matched = sum(1 for kw in expected_keywords if kw in retrieved_lower)
    match_ratio = matched / len(expected_keywords)

    return match_ratio >= 0.6


def run_custom_eval(eval_path: str):
    """Run custom evaluation from eval.txt."""
    print("=" * 70)
    print("  SOP CHATBOT — CUSTOM EVALUATION")
    print("=" * 70)

    # Parse test file
    tests = parse_eval_file(eval_path)
    print(f"\n[FILE] Loaded: {eval_path}")
    print(f"[INFO] Test cases: {len(tests)}")

    # Categorize tests
    in_scope = [t for t in tests if not is_rejection(t["expected"])]
    out_scope = [t for t in tests if is_rejection(t["expected"])]
    print(f"   |-- In-scope (should answer): {len(in_scope)}")
    print(f"   |-- Out-of-scope (should reject): {len(out_scope)}")

    # Load pipeline (no LLM rewrite for evaluation)
    original_rewrite = config.USE_LLM_REWRITE
    config.USE_LLM_REWRITE = False

    pipeline = SOPPipeline()
    pipeline.load()

    print(f"\n[INFO] Threshold: {config.SIMILARITY_THRESHOLD}")

    # ── Run Tests ─────────────────────────────────────────
    passed = 0
    failed = 0
    failures = []

    print("\n" + "─" * 70)
    print("  RUNNING TESTS")
    print("─" * 70)

    start = time.time()

    for i, test in enumerate(tests, 1):
        q = test["question"]
        expected = test["expected"]

        result = pipeline.query(q)
        retrieved = result["answer"]
        score = result["score"]
        rejected = result["rejected"]

        # Check if answer matches expected
        if answer_matches(retrieved, expected):
            passed += 1
            status = "[PASS]"
        else:
            failed += 1
            status = "[FAIL]"
            failures.append({
                "num": i,
                "question": q,
                "expected": expected,
                "got": retrieved,
                "score": score,
                "rejected": rejected,
                "matched_q": result.get("matched_question", "N/A"),
            })

        # Print each test
        print(f"  {status} [{i:3d}] Q: {q}")
        if status == "[FAIL]":
            print(f"         Expected: {expected}")
            print(f"         Got:      {retrieved[:100]}{'...' if len(retrieved) > 100 else ''}")
            print(f"         Score: {score:.4f} | Rejected: {rejected}")

    elapsed = time.time() - start

    # ── Summary ───────────────────────────────────────────
    total = len(tests)
    accuracy = (passed / total) * 100 if total > 0 else 0

    # In-scope accuracy
    in_scope_pass = sum(1 for t in in_scope if t not in [f for f in tests if any(
        ff["question"] == t["question"] for ff in failures)])
    in_scope_acc = (len(in_scope) - sum(1 for f in failures if not is_rejection(f["expected"]))) / len(in_scope) * 100 if in_scope else 0

    # Out-of-scope accuracy
    out_scope_pass = len(out_scope) - sum(1 for f in failures if is_rejection(f["expected"]))
    out_scope_acc = (out_scope_pass / len(out_scope)) * 100 if out_scope else 0

    print("\n" + "=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    print(f"""
  ┌──────────────────────────────────────────────────┐
  │  Overall Accuracy:         {accuracy:6.1f}%                │
  │  In-Scope Accuracy:        {in_scope_acc:6.1f}%                │
  │  Out-of-Scope Rejection:   {out_scope_acc:6.1f}%                │
  │                                                  │
  │  Passed: {passed:3d}/{total:<3d}                               │
  │  Failed: {failed:3d}/{total:<3d}                               │
  │  Time:   {elapsed:.2f}s                                │
  │                                                  │
  │  Threshold: {config.SIMILARITY_THRESHOLD:.2f}                              │
  │  Model: {config.EMBEDDING_MODEL:<20s}           │
  └──────────────────────────────────────────────────┘
""")

    if failures:
        print(f"  [FAIL] DETAILED FAILURES ({len(failures)}):")
        print("  " + "-" * 60)
        for f in failures:
            expect_type = "REJECT" if is_rejection(f["expected"]) else "ANSWER"
            got_type = "REJECTED" if f["rejected"] else "ANSWERED"
            print(f"\n  [{f['num']:3d}] Q: {f['question']}")
            print(f"       Expected ({expect_type}): {f['expected']}")
            print(f"       Got ({got_type}): {f['got'][:120]}{'...' if len(f['got']) > 120 else ''}")
            print(f"       Matched Q: {f['matched_q']}")
            print(f"       Score: {f['score']:.4f}")

    if accuracy == 100.0:
        print("\n  PERFECT SCORE -- All custom tests passed!")
    elif accuracy >= 90.0:
        print("\n  EXCELLENT -- Production ready!")
    elif accuracy >= 75.0:
        print("\n  GOOD -- Minor improvements needed")
    else:
        print("\n  NEEDS IMPROVEMENT -- Review failures above")

    # Restore settings
    config.USE_LLM_REWRITE = original_rewrite

    return {"accuracy": accuracy, "passed": passed, "failed": failed, "total": total}


if __name__ == "__main__":
    # Default eval file path
    if len(sys.argv) > 1:
        eval_file = sys.argv[1]
    else:
        eval_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval.txt")

    if not os.path.exists(eval_file):
        print(f"❌ Eval file not found: {eval_file}")
        sys.exit(1)

    run_custom_eval(eval_file)
