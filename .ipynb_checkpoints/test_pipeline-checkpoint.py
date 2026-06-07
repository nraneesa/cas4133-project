"""
test_pipeline.py
----------------
Mini smoke test — runs the full pipeline on a tiny batch
to verify scorer, optimizer, and opro_loop all work together
BEFORE spending API credits on the full experiment.

Runs:
  - 10 reviews (instead of 100)
  - 3 steps   (instead of 30)
  - Mode C only

Expected cost: ~$0.05 (almost free)
Expected time: ~3-5 minutes

Usage:
    python test_pipeline.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

# ── Check API key first ───────────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not set!")
    print("Run: export OPENAI_API_KEY='sk-...'")
    sys.exit(1)
print(f"✓ API key found: {api_key[:8]}...")


# ── Test 1: Data loading ──────────────────────────────────────────────────────
print("\n── Test 1: Data Loading ──────────────────────────────")
from load_imdb import get_optimization_set, get_test_set

opt_set  = get_optimization_set()
test_set = get_test_set()

# Use only first 10 reviews for smoke test
mini_batch = opt_set[:10]

print(f"✓ Optimization set loaded : {len(opt_set)} reviews")
print(f"✓ Test set loaded         : {len(test_set)} reviews")
print(f"✓ Mini batch              : {len(mini_batch)} reviews")
print(f"  Labels: { {r['label'] for r in mini_batch} }")


# ── Test 2: Scorer ────────────────────────────────────────────────────────────
print("\n── Test 2: Scorer ────────────────────────────────────")
from scorer import score_prompt

instruction = "Classify this movie review as positive or negative."
examples = [
    {"text": opt_set[0]["text"], "label": opt_set[0]["label"]},
    {"text": opt_set[1]["text"], "label": opt_set[1]["label"]},
    {"text": opt_set[2]["text"], "label": opt_set[2]["label"]},
]

result = score_prompt(
    instruction,
    examples,
    reviews=mini_batch,   # only 10 reviews!
    verbose=True
)

print(f"\n✓ Scorer works!")
print(f"  Accuracy : {result['accuracy']:.1%}")
print(f"  Correct  : {result['correct']}/{result['total']}")


# ── Test 3: Optimizer ─────────────────────────────────────────────────────────
print("\n── Test 3: Optimizer ─────────────────────────────────")
from optimizer import generate_new_prompt

# Give it a mock history with one entry (the result from Test 2)
mock_history = [
    {
        "instruction": instruction,
        "examples"   : examples,
        "accuracy"   : result["accuracy"]
    }
]

new_prompt = generate_new_prompt(
    mock_history,
    candidate_reviews=mini_batch,
    verbose=True
)

if new_prompt:
    print(f"\n✓ Optimizer works!")
    print(f"  New instruction : {new_prompt['instruction'][:80]}")
    print(f"  New examples    : {len(new_prompt['examples'])} provided")
    print(f"  Reasoning       : {new_prompt['reasoning']}")
else:
    print("✗ Optimizer failed — check API key and errors above")
    sys.exit(1)


# ── Test 4: Full OPRO Loop (3 steps) ─────────────────────────────────────────
print("\n── Test 4: Full OPRO Loop (3 steps, 10 reviews) ─────")

# Temporarily patch BATCH_SIZE so scorer only uses 10 reviews
import scorer as scorer_module
original_batch_size = scorer_module.BATCH_SIZE
scorer_module.BATCH_SIZE = 10   # override for test

from opro_loop import run_opro
loop_result = run_opro(mode="C", steps=3, verbose=False)

# Restore original batch size
scorer_module.BATCH_SIZE = original_batch_size

print(f"\n✓ OPRO loop works!")
print(f"  Steps run      : {loop_result['total_steps']}")
print(f"  Best accuracy  : {loop_result['best_accuracy']:.1%}")
print(f"  Best prompt    : {loop_result['best_instruction'][:80]}")
print(f"  History entries: {len(loop_result['history'])}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("ALL TESTS PASSED!")
print("="*55)
print(f"  Test 1 (data loading)  : ✓")
print(f"  Test 2 (scorer)        : ✓  accuracy={result['accuracy']:.1%}")
print(f"  Test 3 (optimizer)     : ✓  new prompt generated")
print(f"  Test 4 (full loop)     : ✓  best={loop_result['best_accuracy']:.1%}")
print()
print("Pipeline is working correctly.")
print("You can now run the full experiment:")
print("  python opro_loop.py --mode C --steps 30")
print("  python opro_loop.py --mode B --steps 30")

# Save test results
os.makedirs("results", exist_ok=True)
with open("results/smoke_test_results.json", "w") as f:
    json.dump({
        "scorer_accuracy"  : result["accuracy"],
        "best_loop_accuracy": loop_result["best_accuracy"],
        "best_prompt"      : loop_result["best_instruction"],
        "status"           : "all passed"
    }, f, indent=2)
print("\nTest results saved to results/smoke_test_results.json")
