"""Quick end-to-end test of the SOP pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.pipeline import SOPPipeline

p = SOPPipeline()
p.load()

# Test 1: Exact match
r1 = p.query("What is the PMS value of silver standard?")
print("TEST 1 - Exact match:")
print(f"  Answer: {r1['answer']}")
print(f"  Score: {r1['score']:.4f}")
print(f"  Rejected: {r1['rejected']}")
print()

# Test 2: Paraphrase
r2 = p.query("which pantone for silver?")
print("TEST 2 - Paraphrase:")
print(f"  Answer: {r2['answer']}")
print(f"  Score: {r2['score']:.4f}")
print(f"  Rejected: {r2['rejected']}")
print()

# Test 3: Out-of-scope (must reject)
r3 = p.query("How do I cook pasta?")
print("TEST 3 - Out-of-scope:")
print(f"  Answer: {r3['answer']}")
print(f"  Score: {r3['score']:.4f}")
print(f"  Rejected: {r3['rejected']}")
print()

if not r1["rejected"] and not r2["rejected"] and r3["rejected"]:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED")
