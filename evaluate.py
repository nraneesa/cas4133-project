"""
evaluate.py
-----------
Final test set evaluation — runs the best prompts found by OPRO
on the held-out test set (500 reviews never seen during optimization).

This gives HONEST final accuracy numbers for your slides.

Usage:
    python evaluate.py

What it does:
    1. Loads best prompts from optimization logs
    2. Runs each on the 500-review test set
    3. Saves final comparison table to results/final_evaluation.json
    4. Prints a clean summary table
"""

import os
import sys
import json
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from load_sst5 import get_test_set, get_labels
from scorer   import score_prompt, load_scorer_model, SCORER_MODEL


# ── Config 
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
LABELS       = get_labels()


# ── Load Best Prompt from Logs 
def load_best_prompt(log_path: str) -> dict | None:
    """
    Load the best (instruction, examples) combo from an OPRO log file.

    Returns dict with instruction, examples, accuracy (train), step
    """
    if not os.path.exists(log_path):
        print(f"  [evaluate] Log file not found: {log_path}")
        return None

    with open(log_path) as f:
        logs = json.load(f)

    if not logs:
        print(f"  [evaluate] Log file is empty: {log_path}")
        return None

    best = max(logs, key=lambda x: x["accuracy"])

    return {
        "instruction"   : best["instruction"],
        "examples"      : best.get("examples", []),
        "train_accuracy": best["accuracy"],
        "step"          : best.get("step", 0),
    }


# ── Evaluate Single Condition 
def evaluate_condition(
    name        : str,
    instruction : str,
    examples    : list,
    test_reviews: list,
    verbose     : bool = True
) -> dict:
    """
    Evaluate one (instruction, examples) combo on the full test set.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"  Instruction: {instruction[:70]}...")
        print(f"  Examples   : {len(examples)}")
        print(f"  Test size  : {len(test_reviews)}")
        print(f"{'='*60}")

    result = score_prompt(
        instruction,
        examples,
        reviews=test_reviews,
        verbose=verbose
    )

    return {
        "name"          : name,
        "instruction"   : instruction,
        "examples"      : examples,
        "test_accuracy" : result["accuracy"],
        "correct"       : result["correct"],
        "total"         : result["total"],
    }


# ── Main Evaluation 
def run_evaluation(scorer_model: str = SCORER_MODEL, model_prefix: str = None):
    """
    Run final evaluation for all conditions on the test set.

    Parameters
    ----------
    scorer_model : str
        HuggingFace model name for the scorer
    model_prefix : str
        Prefix used in your result filenames e.g. 'qwen' or 'phi'
        Used to load: {prefix}_baseline_scores.json
                      {prefix}_opro_logs_mode_B.json
                      {prefix}_opro_logs_mode_C_run1.json (picks best run)

    Evaluates:
        1. Baseline A  — zero-shot
        2. Baseline B  — simple instruction
        3. Mode B best — OPRO instruction only
        4. Mode C best — OPRO instruction + examples (our method)
    """
    # Auto-detect prefix from model name if not provided
    if model_prefix is None:
        if "phi" in scorer_model.lower():
            model_prefix = "phi"
        else:
            model_prefix = "qwen"

    print(f"\n{'#'*60}")
    print(f"  FINAL TEST SET EVALUATION")
    print(f"  Scorer model  : {scorer_model}")
    print(f"  Model prefix  : {model_prefix}")
    print(f"  Test set      : 500 reviews (never seen during optimization)")
    print(f"{'#'*60}")

    # ── Load test set 
    test_reviews = get_test_set()
    print(f"\n✓ Test set loaded: {len(test_reviews)} reviews")
    label_dist = {}
    for r in test_reviews:
        label_dist[r["label"]] = label_dist.get(r["label"], 0) + 1
    print(f"  Distribution: {label_dist}")

    # ── Load scorer model 
    load_scorer_model(scorer_model)

    # ── Load baseline prompts 
    # Try model-specific baseline first, fall back to generic
    baseline_path = os.path.join(RESULTS_DIR, f"{model_prefix}_baseline_scores.json")
    if not os.path.exists(baseline_path):
        baseline_path = os.path.join(RESULTS_DIR, "baseline_scores_FINAL.json")
    if not os.path.exists(baseline_path):
        baseline_path = os.path.join(RESULTS_DIR, "baseline_scores.json")

    print(f"\n✓ Loading baselines from: {os.path.basename(baseline_path)}")
    with open(baseline_path) as f:
        baselines = json.load(f)

    baseline_a = next((b for b in baselines if "zero" in b["name"]), None)
    baseline_b = next((b for b in baselines if "simple_instruction" in b["name"]), None)

    # ── Load best OPRO prompts 
    # Mode B
    mode_b_path = os.path.join(RESULTS_DIR, f"{model_prefix}_opro_logs_mode_B.json")
    mode_b_best = load_best_prompt(mode_b_path)
    print(f"✓ Loading Mode B from : {os.path.basename(mode_b_path)}")

    # Mode C — pick best across run1 and run2
    mode_c_run1 = load_best_prompt(os.path.join(RESULTS_DIR, f"{model_prefix}_opro_logs_mode_C_run1.json"))
    mode_c_run2 = load_best_prompt(os.path.join(RESULTS_DIR, f"{model_prefix}_opro_logs_mode_C_run2.json"))

    # Pick whichever run had the higher train accuracy
    candidates = [r for r in [mode_c_run1, mode_c_run2] if r is not None]
    if candidates:
        mode_c_best = max(candidates, key=lambda x: x["train_accuracy"])
        run_num = "run1" if mode_c_best is mode_c_run1 else "run2"
        print(f"✓ Loading Mode C from : {model_prefix}_opro_logs_mode_C_{run_num}.json (best of available runs)")
    else:
        mode_c_best = None
        print(f"✗ No Mode C logs found for prefix: {model_prefix}")

    # ── Define conditions to evaluate 
    conditions = []

    if baseline_a:
        conditions.append({
            "name"       : "A_zero_shot",
            "instruction": baseline_a["instruction"],
            "examples"   : baseline_a.get("examples", []),
        })

    if baseline_b:
        conditions.append({
            "name"       : "B_simple_instruction",
            "instruction": baseline_b["instruction"],
            "examples"   : baseline_b.get("examples", []),
        })

    if mode_b_best:
        conditions.append({
            "name"       : f"Mode_B_OPRO_best (train: {mode_b_best['train_accuracy']:.1%})",
            "instruction": mode_b_best["instruction"],
            "examples"   : [],  # mode B has no examples
        })

    if mode_c_best:
        conditions.append({
            "name"       : f"Mode_C_OPRO_best (train: {mode_c_best['train_accuracy']:.1%})",
            "instruction": mode_c_best["instruction"],
            "examples"   : mode_c_best["examples"],
        })

    if not conditions:
        print("ERROR: No conditions to evaluate — check your results/ folder")
        return
        
    # ── Run evaluation 
    results = []
    for cond in conditions:
        result = evaluate_condition(
            name        = cond["name"],
            instruction = cond["instruction"],
            examples    = cond["examples"],
            test_reviews= test_reviews,
            verbose     = True
        )
        results.append(result)

        # Save after each condition in case of crashes
        _save_results(results, scorer_model)

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(results, scorer_model)

    return results


# ── Save Results 
def _save_results(results: list, scorer_model: str):
    """Save evaluation results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Use model name in filename so Qwen and Phi-3 don't overwrite each other
    model_tag = scorer_model.split("/")[-1].replace("-", "_").lower()
    save_path = os.path.join(RESULTS_DIR, f"final_evaluation_{model_tag}.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  [evaluate] Saved to {save_path}")
    return save_path


# ── Print Summary 
def _print_summary(results: list, scorer_model: str):
    """Print a clean final summary table."""
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — TEST SET (500 reviews)")
    print(f"  Model: {scorer_model}")
    print(f"{'='*60}")
    print(f"  {'Condition':<45} {'Test Acc':>8}")
    print(f"  {'-'*53}")

    for r in results:
        print(f"  {r['name']:<45} {r['test_accuracy']:>8.1%}")

    print(f"  {'-'*53}")

    # Show improvement from Mode B → Mode C
    mode_b = next((r for r in results if "Mode_B" in r["name"]), None)
    mode_c = next((r for r in results if "Mode_C" in r["name"]), None)

    if mode_b and mode_c:
        improvement = mode_c["test_accuracy"] - mode_b["test_accuracy"]
        print(f"\n  Our contribution (Mode C vs Mode B): {improvement:+.1%}")

        if improvement > 0:
            print(f"  ✓ Joint optimization wins on test set!")
        else:
            print(f"  ✗ Joint optimization did not improve on test set")

    print(f"\n  Reminder: optimization set accuracy was measured on 100 reviews")
    print(f"  Test set accuracy is on 500 UNSEEN reviews — this is the honest number")
    print(f"{'='*60}\n")


# ── Main 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["qwen", "phi", "both"],
        default="both",
        help="Which model to evaluate (default: both)"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.model in ["qwen", "both"]:
        run_evaluation(
            scorer_model  = "Qwen/Qwen2.5-1.5B-Instruct",
            model_prefix  = "qwen"
        )

    if args.model in ["phi", "both"]:
        # Free Qwen memory before loading Phi-3
        import gc
        import scorer as scorer_module
        if scorer_module._model is not None:
            del scorer_module._model
            del scorer_module._tokenizer
            scorer_module._model     = None
            scorer_module._tokenizer = None
            scorer_module._loaded_for = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"\nFreed Qwen memory. VRAM now: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        run_evaluation(
            scorer_model  = "microsoft/Phi-3-mini-4k-instruct",
            model_prefix  = "phi"
        )