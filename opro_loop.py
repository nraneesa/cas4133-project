"""
opro_loop.py
------------
The main OPRO loop — connects the Scorer and Optimizer together into
an iterative optimization process.

Configured for:
  - SST-5 dataset (5-class sentiment classification)
  - Local Hugging Face models (no API needed)

Flow:
  1. Generate seed prompts
  2. Score each seed
  3. Send history to Optimizer → get new prompt
  4. Score new prompt
  5. Save to history
  6. Repeat for N steps
  7. Return best prompt found

Usage:
    python opro_loop.py --mode B --steps 30
    python opro_loop.py --mode C --steps 30

Modes:
    B = optimize instruction only       (replicates original OPRO paper)
    C = optimize instruction + examples (our contribution)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from load_sst5   import get_optimization_set
from scorer      import score_prompt, load_scorer_model
from optimizer   import generate_new_prompt, generate_seed_prompts, load_optimizer_model


# ── Config 
DEFAULT_STEPS  = 30       # number of optimization steps
DEFAULT_MODE   = "C"      # B = instruction only, C = instruction + examples
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
LOG_EVERY      = 1        # save logs every N steps


# ── Mode B Helper 
def strip_examples(history: list[dict]) -> list[dict]:
    """
    For Mode B: remove examples from history before passing to optimizer.
    This forces the optimizer to only improve the instruction text,
    replicating the original OPRO paper behavior.
    """
    stripped = []
    for entry in history:
        stripped.append({
            "instruction": entry["instruction"],
            "examples"   : [],        # always empty in mode B
            "accuracy"   : entry["accuracy"]
        })
    return stripped


# ── Logger 
def save_logs(logs: list[dict], mode: str):
    """Save optimization logs to JSON after every step."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"opro_logs_mode_{mode}.json")
    with open(path, "w") as f:
        json.dump(logs, f, indent=2)
    return path


def print_step_summary(step: int, total: int, result: dict, best: dict):
    """Print a clean one-line summary after each step."""
    marker = " ← NEW BEST!" if result["accuracy"] >= best["accuracy"] else ""
    print(
        f"  Step {step:>3}/{total} | "
        f"Acc: {result['accuracy']:.1%} | "
        f"Best: {best['accuracy']:.1%}"
        f"{marker}"
    )


# ── Main OPRO Loop 
def run_opro(
    mode    : str  = DEFAULT_MODE,
    steps   : int  = DEFAULT_STEPS,
    verbose : bool = True
) -> dict:
    """
    Run the full OPRO optimization loop.

    Parameters
    ----------
    mode : str
        'B' = optimize instruction only (replicates paper)
        'C' = optimize instruction + examples (our contribution)
    steps : int
        Number of optimization iterations to run.
    verbose : bool
        Print detailed progress.

    Returns
    -------
    dict with best result and full history.
    """
    start_time = datetime.now()
    reviews    = get_optimization_set()

    print(f"\n{'='*60}")
    print(f"  OPRO Loop — Mode {mode}")
    print(f"  Mode description: {'Instruction only' if mode == 'B' else 'Instruction + Examples'}")
    print(f"  Steps: {steps}")
    print(f"  Started: {start_time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── Load models upfront so they don't reload mid-loop 
    print("── Loading models ────────────────────────────────────")
    load_scorer_model()
    load_optimizer_model()

    # ── Step 0: Generate and score seed prompts 
    print("\n── Phase 0: Scoring seed prompts ─────────────────────")
    seeds   = generate_seed_prompts(3)
    history = []

    for i, seed in enumerate(seeds, 1):
        print(f"\nSeed {i}/3:")

        # In mode B, seeds have no examples
        if mode == "B":
            seed["examples"] = []

        result = score_prompt(
            seed["instruction"],
            seed["examples"],
            reviews,
            verbose=verbose
        )

        entry = {
            "step"       : 0,
            "seed"       : i,
            "instruction": seed["instruction"],
            "examples"   : seed["examples"],
            "accuracy"   : result["accuracy"],
            "correct"    : result["correct"],
            "total"      : result["total"],
            "timestamp"  : datetime.now().isoformat(),
            "reasoning"  : "seed prompt"
        }
        history.append(entry)

    # Track best result so far
    best = max(history, key=lambda x: x["accuracy"])
    print(f"\n── Seeds complete. Best seed accuracy: {best['accuracy']:.1%}")
    save_logs(history, mode)

    # ── Steps 1-N: Optimize 
    print(f"\n── Starting optimization ({steps} steps) ──────────────")

    for step in range(1, steps + 1):

        # Build history for optimizer
        # Mode B: strip examples so optimizer only improves instruction
        # Mode C: pass full history including examples
        optimizer_history = strip_examples(history) if mode == "B" else history

        # Ask optimizer to generate better prompt
        new_prompt = generate_new_prompt(
            optimizer_history,
            candidate_reviews=reviews,
            verbose=False
        )

        if new_prompt is None:
            print(f"  Step {step:>3} | Optimizer failed — skipping this step")
            continue

        # In mode B, ignore any examples the optimizer suggests
        if mode == "B":
            new_prompt["examples"] = []

        # Score the new prompt
        result = score_prompt(
            new_prompt["instruction"],
            new_prompt["examples"],
            reviews,
            verbose=False
        )

        # Update best
        if result["accuracy"] >= best["accuracy"]:
            best = {
                "instruction": new_prompt["instruction"],
                "examples"   : new_prompt["examples"],
                "accuracy"   : result["accuracy"],
            }

        # Log this step
        entry = {
            "step"       : step,
            "instruction": new_prompt["instruction"],
            "examples"   : new_prompt["examples"],
            "accuracy"   : result["accuracy"],
            "correct"    : result["correct"],
            "total"      : result["total"],
            "timestamp"  : datetime.now().isoformat(),
            "reasoning"  : new_prompt.get("reasoning", "")
        }
        history.append(entry)

        # Print and save
        print_step_summary(step, steps, result, best)
        if step % LOG_EVERY == 0:
            save_logs(history, mode)

    # ── Final save 
    save_logs(history, mode)
    elapsed = (datetime.now() - start_time).seconds // 60

    print(f"\n{'='*60}")
    print(f"  OPRO Complete — Mode {mode}")
    print(f"  Total steps  : {steps}")
    print(f"  Time elapsed : ~{elapsed} minutes")
    print(f"  Best accuracy: {best['accuracy']:.1%}")
    print(f"  Best prompt  : {best['instruction'][:70]}...")
    print(f"  Log saved to : results/phi_opro_logs_mode_{mode}.json")
    print(f"{'='*60}\n")

    return {
        "best_instruction": best["instruction"],
        "best_examples"   : best.get("examples", []),
        "best_accuracy"   : best["accuracy"],
        "history"         : history,
        "mode"            : mode,
        "total_steps"     : steps
    }


# ── Run All Conditions 
def run_all_conditions(steps: int = DEFAULT_STEPS):
    """
    Run both Mode B and Mode C, save combined summary.

    Note: Condition A (zero-shot baseline) is handled in scorer.py
    """
    results = {}

    for mode in ["B", "C"]:
        print(f"\n{'#'*60}")
        print(f"# Running Condition {mode}")
        print(f"{'#'*60}")
        result = run_opro(mode=mode, steps=steps)
        results[mode] = result
        print("Pausing 10 seconds before next condition...")
        time.sleep(10)

    # Save combined summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {
        "B": {
            "best_instruction": results["B"]["best_instruction"],
            "best_accuracy"   : results["B"]["best_accuracy"],
            "mode_description": "Instruction only (replicates paper)"
        },
        "C": {
            "best_instruction": results["C"]["best_instruction"],
            "best_examples"   : results["C"]["best_examples"],
            "best_accuracy"   : results["C"]["best_accuracy"],
            "mode_description": "Instruction + Examples (our contribution)"
        }
    }
    summary_path = os.path.join(RESULTS_DIR, "opro_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ALL CONDITIONS COMPLETE")
    print(f"{'='*60}")
    print(f"  Mode B (instruction only)      : {results['B']['best_accuracy']:.1%}")
    print(f"  Mode C (instruction + examples): {results['C']['best_accuracy']:.1%}")
    improvement = results["C"]["best_accuracy"] - results["B"]["best_accuracy"]
    print(f"  Improvement from our method    : {improvement:+.1%}")
    print(f"\n  Summary saved to: {summary_path}")

    return results


# ── Main 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OPRO optimization loop")
    parser.add_argument(
        "--mode",
        choices=["B", "C", "all"],
        default="C",
        help="B = instruction only | C = instruction+examples | all = run both"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of optimization steps (default: {DEFAULT_STEPS})"
    )
    args = parser.parse_args()

    # Check GPU available
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        print("This script needs a GPU. Run on Vessel.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.mode == "all":
        run_all_conditions(steps=args.steps)
    else:
        run_opro(mode=args.mode, steps=args.steps)