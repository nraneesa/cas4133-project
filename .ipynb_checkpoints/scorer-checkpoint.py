"""
scorer.py
---------
The Scorer LLM — evaluates how good a (instruction, examples) combo is
by testing it against the optimization set and returning an accuracy score.

This version is configured for SST-5 (5-class fine-grained sentiment).

Usage:
    from scorer import score_prompt

    instruction = "Classify this movie review's sentiment on a 5-point scale."
    examples = [
        {"text": "An absolute masterpiece!",   "label": "very_positive"},
        {"text": "Pretty good film.",          "label": "positive"},
        {"text": "It was okay.",               "label": "neutral"},
    ]

    result = score_prompt(instruction, examples)
    print(result["accuracy"])
"""

import os
import sys
import json
import time

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from load_sst5 import get_optimization_set, get_labels


# ── Config ────────────────────────────────────────────────────────────────────
SCORER_MODEL   = "gpt-3.5-turbo"      # weaker model — gives OPRO room to improve
MAX_RETRIES    = 3
RETRY_DELAY    = 2
BATCH_SIZE     = 100

# Valid labels for SST-5
LABELS         = get_labels()         # ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
LABEL_LIST_STR = ", ".join(f"'{l}'" for l in LABELS)


# ── Client ────────────────────────────────────────────────────────────────────
def get_client():
    """Return OpenAI client. Called lazily so import works without API key."""
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── Prompt Builder ────────────────────────────────────────────────────────────
def build_prompt(instruction: str, examples: list[dict], review_text: str) -> str:
    """Build the full prompt sent to the Scorer LLM for a single review."""
    lines = []

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
    lines.append(f"Label (respond with ONLY one of: {LABEL_LIST_STR}):")

    return "\n".join(lines)


# ── Response Parser ───────────────────────────────────────────────────────────
def parse_label(raw: str) -> str:
    """
    Parse LLM response into one of the 5 valid labels.
    Returns 'unknown' if no valid label is detected.
    """
    raw = raw.strip().lower()
    # Normalize separators
    cleaned = raw.replace(" ", "_").replace("-", "_")

    # Direct match (check very_* first so 'positive' doesn't match 'very_positive')
    for label in ["very_negative", "very_positive", "negative", "positive", "neutral"]:
        if label in cleaned:
            return label

    # Soft fallback
    if "very" in raw and ("pos" in raw or "good" in raw):
        return "very_positive"
    if "very" in raw and ("neg" in raw or "bad" in raw):
        return "very_negative"
    if "pos" in raw or "good" in raw:
        return "positive"
    if "neg" in raw or "bad" in raw:
        return "negative"
    if "neutral" in raw or "mixed" in raw or "okay" in raw:
        return "neutral"

    return "unknown"


# ── Single Review Classifier ──────────────────────────────────────────────────
def classify_review(instruction: str, examples: list[dict], review_text: str) -> str:
    """Ask the LLM to classify a single review on the 5-point scale."""
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
                            f"You are a fine-grained sentiment classifier. "
                            f"You must respond with ONLY one label from this list: "
                            f"{LABEL_LIST_STR}. No explanation."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=10,
                temperature=0.0,
            )

            raw    = response.choices[0].message.content
            label  = parse_label(raw)

            if label == "unknown":
                print(f"    [scorer] Unexpected response: '{raw.strip()}' — marking as unknown")

            return label

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
    """Score a (instruction, examples) combo on the optimization set."""
    if reviews is None:
        reviews = get_optimization_set()

    reviews = reviews[:BATCH_SIZE]

    if verbose:
        print(f"\n── Scoring prompt ────────────────────────────────────")
        print(f"  Instruction : {instruction[:80]}{'...' if len(instruction) > 80 else ''}")
        print(f"  Examples    : {len(examples)} provided")
        print(f"  Reviews     : {len(reviews)} to score")
        print(f"  Model       : {SCORER_MODEL}")
        print(f"  Task        : SST-5 (5-class sentiment)")
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


# ── Baseline Scorer ───────────────────────────────────────────────────────────
def score_baseline_prompts(save_path: str = "results/baseline_scores.json"):
    """Score 3 baseline prompts for the SST-5 task."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    reviews = get_optimization_set()

    # Pick 5 diverse examples (one per class if possible)
    examples_by_class = {}
    for r in reviews:
        if r["label"] not in examples_by_class:
            examples_by_class[r["label"]] = r
        if len(examples_by_class) == 5:
            break
    sample_examples = list(examples_by_class.values())[:5]

    baselines = [
        {
            "name"       : "A_zero_shot",
            "instruction": "Classify this movie review.",
            "examples"   : [],
            "description": "No instruction, no examples (zero-shot baseline)"
        },
        {
            "name"       : "B_simple_instruction",
            "instruction": (
                f"Classify this movie review's sentiment on a 5-point scale. "
                f"Choose from: {LABEL_LIST_STR}."
            ),
            "examples"   : [],
            "description": "Simple instruction, no examples"
        },
        {
            "name"       : "C_simple_with_examples",
            "instruction": (
                f"Classify this movie review's sentiment on a 5-point scale. "
                f"Choose from: {LABEL_LIST_STR}."
            ),
            "examples"   : sample_examples,
            "description": "Simple instruction + 5 examples (one per class)"
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
    print("BASELINE SUMMARY (SST-5)")
    print(f"{'='*55}")
    for r in results:
        print(f"  {r['name']:<30} {r['accuracy']:.1%}")
    print(f"\nSaved to {save_path}")
    print()
    print(f"Expected range with gpt-3.5-turbo: 40-55%")
    print(f"If baselines are 40-55%, OPRO has plenty of room to improve!")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set!")
        print("Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print("OpenAI API key found.")
    print(f"Scorer model: {SCORER_MODEL}")
    print(f"Task: SST-5 (5-class sentiment classification)")
    print(f"Labels: {LABELS}")
    print("\nRunning baseline scoring...\n")
    score_baseline_prompts()
