"""
scorer.py
---------
The Scorer LLM — evaluates how good a (instruction, examples) combo is
by testing it against the optimization set and returning an accuracy score.

This is the "objective function" of the OPRO loop.

Usage:
    from scorer import score_prompt

    instruction = "Classify this movie review as positive or negative."
    examples = [
        {"text": "Loved every minute!", "label": "positive"},
        {"text": "Terrible film.",      "label": "negative"},
        {"text": "A true masterpiece.", "label": "positive"},
    ]

    result = score_prompt(instruction, examples)
    print(result["accuracy"])   # e.g. 0.82
"""

import os
import sys
import json
import time

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from load_imdb import get_optimization_set


# ── Config ────────────────────────────────────────────────────────────────────
SCORER_MODEL   = "gpt-4o-mini"       # cheap + fast, good enough for scoring
MAX_RETRIES    = 3                    # retry on API errors
RETRY_DELAY    = 2                    # seconds between retries
BATCH_SIZE     = 100                  # how many reviews to score (full opt set)


# ── Client ────────────────────────────────────────────────────────────────────
def get_client():
    """Return OpenAI client. Called lazily so import works without API key."""
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(instruction: str, examples: list[dict], review_text: str) -> str:
    """
    Build the full prompt sent to the Scorer LLM for a single review.

    Structure:
        [instruction]

        Example 1:
        Review: ...
        Label: positive/negative

        ... more examples ...

        Now classify this review:
        Review: ...
        Label:
    """
    lines = []

    # Instruction
    lines.append(instruction.strip())
    lines.append("")

    # Few-shot examples
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Review: {ex['text'].strip()}")
        lines.append(f"Label: {ex['label'].strip()}")
        lines.append("")

    # The review to classify
    lines.append("Now classify this review.")
    lines.append(f"Review: {review_text.strip()}")
    lines.append("Label (respond with ONLY 'positive' or 'negative'):")

    return "\n".join(lines)


# ── Single Review Classifier ──────────────────────────────────────────────────
def classify_review(instruction: str, examples: list[dict], review_text: str) -> str:
    """
    Ask the LLM to classify a single review.

    Returns 'positive', 'negative', or 'unknown' on failure.
    """
    prompt = build_prompt(instruction, examples, review_text)

    client = get_client()
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=SCORER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a sentiment classifier. "
                            "You must respond with ONLY one word: "
                            "'positive' or 'negative'. No explanation."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=5,        # only need one word
                temperature=0.0,     # deterministic — same input = same output
            )

            raw = response.choices[0].message.content.strip().lower()

            # Clean up response — sometimes model adds punctuation
            if "positive" in raw:
                return "positive"
            elif "negative" in raw:
                return "negative"
            else:
                print(f"    [scorer] Unexpected response: '{raw}' — marking as unknown")
                return "unknown"

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [scorer] API error (attempt {attempt+1}): {e} — retrying...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [scorer] Failed after {MAX_RETRIES} attempts: {e}")
                return "unknown"


# ── Main Scorer ───────────────────────────────────────────────────────────────
def score_prompt(
    instruction: str,
    examples: list[dict],
    reviews: list[dict] | None = None,
    verbose: bool = True
) -> dict:
    """
    Score a (instruction, examples) combo on the optimization set.

    Parameters
    ----------
    instruction : str
        The instruction text to prepend to every review.
    examples : list of dict
        Few-shot examples, each with 'text' and 'label' keys.
        Pass an empty list [] for zero-shot (no examples).
    reviews : list of dict, optional
        Reviews to score against. Defaults to full optimization set.
    verbose : bool
        Print progress as scoring runs.

    Returns
    -------
    dict with keys:
        instruction  : str    — the instruction that was scored
        examples     : list   — the examples that were scored
        accuracy     : float  — fraction correct (0.0 – 1.0)
        correct      : int    — number of correct predictions
        total        : int    — total reviews scored
        predictions  : list   — per-review results for debugging
    """
    if reviews is None:
        reviews = get_optimization_set()

    reviews = reviews[:BATCH_SIZE]

    if verbose:
        print(f"\n── Scoring prompt ────────────────────────────────────")
        print(f"  Instruction : {instruction[:80]}{'...' if len(instruction) > 80 else ''}")
        print(f"  Examples    : {len(examples)} provided")
        print(f"  Reviews     : {len(reviews)} to score")
        print(f"  Model       : {SCORER_MODEL}")
        print()

    correct     = 0
    predictions = []

    for i, review in enumerate(reviews):
        true_label  = review["label"]
        pred_label  = classify_review(instruction, examples, review["text"])
        is_correct  = pred_label == true_label

        if is_correct:
            correct += 1

        predictions.append({
            "index"     : i,
            "text"      : review["text"][:100],
            "true_label": true_label,
            "pred_label": pred_label,
            "correct"   : is_correct
        })

        if verbose and (i + 1) % 10 == 0:
            running_acc = correct / (i + 1)
            print(f"  [{i+1:>3}/{len(reviews)}] Running accuracy: {running_acc:.1%}")

    accuracy = correct / len(reviews) if reviews else 0.0

    if verbose:
        print(f"\n── Result ────────────────────────────────────────────")
        print(f"  Accuracy : {accuracy:.1%}  ({correct}/{len(reviews)} correct)")

    return {
        "instruction": instruction,
        "examples"   : examples,
        "accuracy"   : accuracy,
        "correct"    : correct,
        "total"      : len(reviews),
        "predictions": predictions
    }


# ── Baseline Scorer (no API needed) ───────────────────────────────────────────
def score_baseline_prompts(save_path: str = "results/baseline_scores.json"):
    """
    Score 3 baseline prompts to establish reference points before OPRO runs.

    Condition A: zero-shot, no instruction, no examples
    Condition B: simple instruction, no examples
    Condition C: simple instruction + 3 random examples
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    reviews  = get_optimization_set()
    baselines = [
        {
            "name"       : "A_zero_shot",
            "instruction": "Classify this movie review.",
            "examples"   : [],
            "description": "No instruction, no examples (zero-shot baseline)"
        },
        {
            "name"       : "B_simple_instruction",
            "instruction": "Classify this movie review as positive or negative.",
            "examples"   : [],
            "description": "Simple instruction only, no examples"
        },
        {
            "name"       : "C_simple_with_examples",
            "instruction": "Classify this movie review as positive or negative.",
            "examples"   : [
                {"text": reviews[0]["text"], "label": reviews[0]["label"]},
                {"text": reviews[1]["text"], "label": reviews[1]["label"]},
                {"text": reviews[2]["text"], "label": reviews[2]["label"]},
            ],
            "description": "Simple instruction + 3 random examples"
        },
    ]

    results = []
    for b in baselines:
        print(f"\n{'='*55}")
        print(f"Scoring baseline: {b['name']}")
        print(f"Description     : {b['description']}")

        result = score_prompt(b["instruction"], b["examples"], reviews)
        result["name"]        = b["name"]
        result["description"] = b["description"]
        results.append(result)

    # Save baseline scores
    save_data = [
        {k: v for k, v in r.items() if k != "predictions"}
        for r in results
    ]
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n{'='*55}")
    print("BASELINE SUMMARY")
    print(f"{'='*55}")
    for r in results:
        print(f"  {r['name']:<30} {r['accuracy']:.1%}")
    print(f"\nSaved to {save_path}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke test with a single review (no API key needed check)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set!")
        print("Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print("OpenAI API key found.")
    print("Running baseline scoring...\n")
    score_baseline_prompts()
