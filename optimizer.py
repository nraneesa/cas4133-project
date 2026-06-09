"""
optimizer.py
------------
The Optimizer LLM — looks at the history of all past (instruction, examples,
score) triplets and generates a new, hopefully better combination.

This version is configured for SST-5 (5-class fine-grained sentiment).
"""

import os
import sys
import json
import time
import random

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from load_sst5 import get_optimization_set, get_labels


# ── Config ────────────────────────────────────────────────────────────────────
OPTIMIZER_MODEL   = "gpt-4o-mini"   # smarter model for generating better prompts
MAX_RETRIES       = 3
RETRY_DELAY       = 2
NUM_EXAMPLES      = 5                # one example per class for 5-class task
MAX_HISTORY_SHOWN = 8

LABELS         = get_labels()
LABEL_LIST_STR = ", ".join(f"'{l}'" for l in LABELS)


# ── Client ────────────────────────────────────────────────────────────────────
def get_client():
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── History Formatter ─────────────────────────────────────────────────────────
def format_history(history: list[dict]) -> str:
    """Format past (instruction, examples, score) history sorted low→high."""
    sorted_history = sorted(history, key=lambda x: x["accuracy"])
    shown = sorted_history[-MAX_HISTORY_SHOWN:]

    lines = []
    for i, entry in enumerate(shown, 1):
        acc_pct = f"{entry['accuracy']:.1%}"
        lines.append(f"Attempt {i}:")
        lines.append(f"  Instruction : {entry['instruction']}")
        lines.append(f"  Examples    :")
        for j, ex in enumerate(entry.get("examples", []), 1):
            lines.append(f"    {j}. [{ex['label']}] {ex['text'][:80]}")
        lines.append(f"  Score       : {acc_pct}")
        lines.append("")

    return "\n".join(lines)


# ── Optimizer Prompt Builder ──────────────────────────────────────────────────
def build_optimizer_prompt(history: list[dict], candidate_reviews: list[dict]) -> str:
    """Build the meta-prompt sent to the Optimizer LLM."""
    history_text = format_history(history)

    # Show 25 candidate reviews (5 per class) so optimizer has variety
    candidates_by_class = {label: [] for label in LABELS}
    for r in candidate_reviews:
        if r["label"] in candidates_by_class and len(candidates_by_class[r["label"]]) < 5:
            candidates_by_class[r["label"]].append(r)
    all_candidates = []
    for label in LABELS:
        all_candidates.extend(candidates_by_class[label])
    random.shuffle(all_candidates)

    candidate_text = ""
    for i, r in enumerate(all_candidates, 1):
        candidate_text += f"  {i}. [{r['label']}] {r['text'][:100]}\n"

    prompt = f"""You are an expert prompt engineer optimizing a fine-grained sentiment classifier for movie reviews.

TASK:
The classifier reads a movie review and must output EXACTLY one of these 5 labels:
  {LABEL_LIST_STR}

This is a HARD task — distinguishing between adjacent classes (e.g., 'positive' vs 'very_positive')
requires careful prompt engineering and well-chosen examples.

Your goal is to find the BEST combination of:
  1. An instruction that tells the LLM how to classify reviews
  2. Five few-shot examples (ideally one per class) that demonstrate clear boundaries

PAST ATTEMPTS (sorted from worst to best score):
{history_text}
The best score so far is {max(e['accuracy'] for e in history):.1%}.

AVAILABLE REVIEWS TO USE AS EXAMPLES:
{candidate_text}
WHAT TO DO:
Study the pattern — what made higher-scoring attempts better than lower ones?
Then generate a NEW instruction and NEW set of 5 examples that will score HIGHER.

Rules:
- The instruction must be clear about the 5-class scale and what distinguishes each class
- Choose examples that cover ALL 5 classes (one per class is ideal)
- Pick examples that clearly show the boundary between adjacent classes
- Do NOT reuse an instruction that already appears in the history above
- Examples must come from the available reviews listed above
- Use the exact label spelling: {LABEL_LIST_STR}

Respond with ONLY a valid JSON object in this exact format:
{{
  "instruction": "your new instruction text here",
  "examples": [
    {{"text": "exact review text from the list above", "label": "one of the 5 valid labels"}},
    {{"text": "exact review text from the list above", "label": "one of the 5 valid labels"}},
    {{"text": "exact review text from the list above", "label": "one of the 5 valid labels"}},
    {{"text": "exact review text from the list above", "label": "one of the 5 valid labels"}},
    {{"text": "exact review text from the list above", "label": "one of the 5 valid labels"}}
  ],
  "reasoning": "one sentence explaining why you think this will score higher"
}}

JSON only. No markdown. No explanation outside the JSON."""

    return prompt


# ── Response Parser ───────────────────────────────────────────────────────────
def parse_optimizer_response(raw: str) -> dict | None:
    """Parse the optimizer's JSON response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)

        if "instruction" not in parsed:
            print("    [optimizer] Missing 'instruction' in response")
            return None
        if "examples" not in parsed or len(parsed["examples"]) < 1:
            print("    [optimizer] Missing or empty 'examples' in response")
            return None

        # Validate and normalize each example's label
        valid_examples = []
        for ex in parsed["examples"]:
            if "text" not in ex or "label" not in ex:
                print(f"    [optimizer] Skipping malformed example: {ex}")
                continue
            label = ex["label"].strip().lower().replace(" ", "_").replace("-", "_")
            if label not in LABELS:
                # Try to coerce to closest valid label
                if "very" in label and ("pos" in label or "good" in label):
                    label = "very_positive"
                elif "very" in label and ("neg" in label or "bad" in label):
                    label = "very_negative"
                elif "pos" in label:
                    label = "positive"
                elif "neg" in label:
                    label = "negative"
                elif "neutral" in label:
                    label = "neutral"
                else:
                    print(f"    [optimizer] Invalid label '{ex['label']}' — skipping example")
                    continue
            ex["label"] = label
            valid_examples.append(ex)

        if len(valid_examples) == 0:
            print("    [optimizer] No valid examples after parsing")
            return None

        return {
            "instruction": parsed["instruction"].strip(),
            "examples"   : valid_examples[:NUM_EXAMPLES],
            "reasoning"  : parsed.get("reasoning", "")
        }

    except json.JSONDecodeError as e:
        print(f"    [optimizer] JSON parse error: {e}")
        print(f"    [optimizer] Raw response was: {raw[:200]}")
        return None


# ── Main Optimizer ────────────────────────────────────────────────────────────
def generate_new_prompt(
    history: list[dict],
    candidate_reviews: list[dict] | None = None,
    verbose: bool = True
) -> dict | None:
    """Given past history, generate a new (instruction, examples) combo."""
    if candidate_reviews is None:
        candidate_reviews = get_optimization_set()

    optimizer_prompt = build_optimizer_prompt(history, candidate_reviews)
    client           = get_client()

    if verbose:
        best_so_far = max(e["accuracy"] for e in history)
        print(f"\n── Generating new prompt ─────────────────────────────")
        print(f"  History size  : {len(history)} attempts")
        print(f"  Best so far   : {best_so_far:.1%}")
        print(f"  Model         : {OPTIMIZER_MODEL}")

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPTIMIZER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert prompt engineer. "
                            "You always respond with valid JSON only. "
                            "No markdown, no explanation outside JSON."
                        )
                    },
                    {
                        "role": "user",
                        "content": optimizer_prompt
                    }
                ],
                max_tokens=1200,
                temperature=0.8,
            )

            raw    = response.choices[0].message.content
            result = parse_optimizer_response(raw)

            if result is not None:
                if verbose:
                    print(f"  New instruction : {result['instruction'][:80]}...")
                    print(f"  Reasoning       : {result['reasoning']}")
                return result
            else:
                if attempt < MAX_RETRIES - 1:
                    print(f"    [optimizer] Parse failed, retrying ({attempt+2}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY)

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"    [optimizer] API error (attempt {attempt+1}): {e} — retrying...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    [optimizer] Failed after {MAX_RETRIES} attempts: {e}")

    return None


# ── Seed Prompt Generator ─────────────────────────────────────────────────────
def generate_seed_prompts(n: int = 3) -> list[dict]:
    """Generate n diverse seed prompts to kick off the OPRO loop."""
    reviews = get_optimization_set()

    # Group reviews by class
    by_class = {label: [] for label in LABELS}
    for r in reviews:
        if r["label"] in by_class:
            by_class[r["label"]].append(r)

    # Build 3 seed prompts with one example per class
    def pick_one_per_class(offset):
        return [
            {"text": by_class[label][offset % len(by_class[label])]["text"],
             "label": label}
            for label in LABELS
        ]

    seeds = [
        {
            "instruction": (
                f"Classify this movie review's sentiment. "
                f"Choose from: {LABEL_LIST_STR}."
            ),
            "examples": pick_one_per_class(0)
        },
        {
            "instruction": (
                "Read the movie review below carefully. "
                "Rate its sentiment on a 5-point scale where "
                "'very_negative' means strongly disliked, 'negative' means disliked, "
                "'neutral' means mixed or no clear opinion, 'positive' means liked, "
                "and 'very_positive' means loved. "
                "Reply with one label only."
            ),
            "examples": pick_one_per_class(1)
        },
        {
            "instruction": (
                f"What is the sentiment of this movie review? "
                f"Pick exactly one: {LABEL_LIST_STR}. "
                f"Be precise — distinguish between 'positive' and 'very_positive', "
                f"and between 'negative' and 'very_negative'."
            ),
            "examples": pick_one_per_class(2)
        },
    ]

    return seeds[:n]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set!")
        sys.exit(1)

    print("Testing optimizer for SST-5...\n")

    mock_history = [
        {
            "instruction": f"Classify this review. Choose from: {LABEL_LIST_STR}.",
            "examples"   : [
                {"text": "Loved every minute!",          "label": "very_positive"},
                {"text": "Pretty good film.",            "label": "positive"},
                {"text": "It was okay.",                  "label": "neutral"},
                {"text": "Boring and slow.",              "label": "negative"},
                {"text": "Worst movie of the year.",     "label": "very_negative"},
            ],
            "accuracy": 0.45
        },
        {
            "instruction": "Rate the sentiment on a 5-point scale.",
            "examples"   : [
                {"text": "A masterpiece.",                "label": "very_positive"},
                {"text": "Solid entertainment.",          "label": "positive"},
                {"text": "Mixed feelings.",               "label": "neutral"},
                {"text": "Disappointing.",                "label": "negative"},
                {"text": "Truly awful.",                  "label": "very_negative"},
            ],
            "accuracy": 0.52
        },
    ]

    result = generate_new_prompt(mock_history, verbose=True)

    if result:
        print("\n── Generated Prompt ──────────────────────────────────")
        print(f"Instruction : {result['instruction']}")
        print(f"Examples    :")
        for i, ex in enumerate(result["examples"], 1):
            print(f"  {i}. [{ex['label']}] {ex['text'][:80]}")
        print(f"Reasoning   : {result['reasoning']}")
        print("\noptimizer.py is working correctly!")
