"""
test_pipeline.py
----------------
Mini smoke test — runs the full pipeline on a tiny batch to verify
scorer, optimizer, and opro_loop all work together with SST-5.

Runs:
  - 10 reviews (instead of 100)
  - 3 steps   (instead of 30)
  - Mode C only

Expected cost: ~$0.05
Expected time: ~3-5 minutes
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

# ── Check API key first ───────────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not set!")
    sys.exit(1)
print(f"✓ API key found: {api_key[:8]}...")


# ── Test 1: Data loading ──────────────────────────────────────────────────────
print("\n── Test 1: Data Loading ──────────────────────────────")
from load_sst5 import get_optimization_set, get_test_set, get_labels

opt_set  = get_optimization_set()
test_set = get_test_set()
labels   = get_labels()

mini_batch = opt_set[:10]

print(f"✓ Optimization set : {len(opt_set)} samples")
print(f"✓ Test set         : {len(test_set)} samples")
print(f"✓ Mini batch       : {len(mini_batch)} samples")
print(f"✓ Labels           : {labels}")
print(f"  Mini batch label distribution: { {l: sum(1 for r in mini_batch if r['label']==l) for l in labels} }")


# ── Test 2: Scorer ────────────────────────────────────────────────────────────
print("\n── Test 2: Scorer ────────────────────────────────────")
from scorer import score_prompt, LABEL_LIST_STR

# Pick one example per class if available
examples = []
seen = set()
for r in opt_set:
    if r["label"] not in seen:
        examples.append({"text": r["text"], "label": r["label"]})
        seen.add(r["label"])
    if len(examples) == 5:
        break

instruction = f"Classify this movie review's sentiment. Choose from: {LABEL_LIST_STR}."

result = score_prompt(
    instruction,
    examples,
    reviews=mini_batch,
    verbose=True
)

print(f"\n✓ Scorer works! Accuracy: {result['accuracy']:.1%}")


# ── Test 3: Optimizer ─────────────────────────────────────────────────────────
print("\n── Test 3: Optimizer ─────────────────────────────────")
from optimizer import generate_new_prompt

mock_history = [
    {
        "instruction": instruction,
        "examples"   : examples,
        "accuracy"   : result["accuracy"]
    }
]

new_prompt = generate_new_prompt(
    mock_history,
    candidate_reviews=opt_set,
    verbose=True
)

if new_prompt:
    print(f"\n✓ Optimizer works!")
    print(f"  New instruction : {new_prompt['instruction'][:80]}")
    print(f"  New examples    : {len(new_prompt['examples'])} provided")
else:
    print("✗ Optimizer failed")
    sys.exit(1)


# ── Test 4: Full OPRO Loop ────────────────────────────────────────────────────
print("\n── Test 4: Full OPRO Loop (3 steps, 10 reviews) ─────")

import scorer as scorer_module
original_batch_size = scorer_module.BATCH_SIZE
scorer_module.BATCH_SIZE = 10

from opro_loop import run_opro
loop_result = run_opro(mode="C", steps=3, verbose=False)

scorer_module.BATCH_SIZE = original_batch_size

print(f"\n✓ OPRO loop works!")
print(f"  Best accuracy: {loop_result['best_accuracy']:.1%}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("ALL TESTS PASSED!")
print("="*55)
print(f"  Test 2 (scorer 10 reviews) : {result['accuracy']:.1%}")
print(f"  Test 4 (loop 3 steps)      : {loop_result['best_accuracy']:.1%}")
print()
print("Pipeline works for SST-5.")
print("Next: run full baselines with `python scorer.py`")
