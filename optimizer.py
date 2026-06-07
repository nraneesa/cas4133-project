"""
optimizer.py
------------
The Optimizer LLM — looks at the history of all past (instruction, examples,
score) sets and generates a new, hopefully better combination.

This is the "search engine" of the OPRO loop.

Usage:
    from optimizer import generate_new_prompt

    history = [
        {
            "instruction": "Classify this review.",
            "examples": [...],
            "accuracy": 0.72
        },
        {
            "instruction": "Determine if this movie review is positive or negative.",
            "examples": [...],
            "accuracy": 0.81
        },
    ]

    result = generate_new_prompt(history)
    print(result["instruction"])   # new instruction text
    print(result["examples"])      # new list of 3 examples
"""

import os
import sys
import json
import time
import random

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from load_imdb import get_optimization_set


# ── Config 
OPTIMIZER_MODEL  = "gpt-4o-mini"   # can upgrade to gpt-4o for better suggestions
MAX_RETRIES      = 3
RETRY_DELAY      = 2
NUM_EXAMPLES     = 3               # how many few-shot examples to optimize
# How many past results to show the optimizer
MAX_HISTORY_SHOWN = 8


# ── Client 
def get_client():
    """Return OpenAI client. Called lazily so import works without API key."""
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── History Formatter 
# format all the past sets (instructions, examples and score)
# sorted the sets from worst to best
def format_history(history: list[dict]) -> str:
    """
    Format the past (instruction, examples, score) history for the optimizer.

    Shows results sorted lowest → highest so the optimizer can clearly
    see the trend of what works better.
    """
    # Sort by accuracy ascending (worst → best)
    sorted_history = sorted(history, key=lambda x: x["accuracy"])

    # Only show the last MAX_HISTORY_SHOWN to avoid bloating the prompt
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


# ── Optimizer Prompt Builder
def build_optimizer_prompt(history: list[dict], candidate_reviews: list[dict]) -> str:
    """
    Build the meta-prompt sent to the Optimizer LLM.

    The optimizer sees:
    1. Task description
    2. History of past attempts sorted by score (low → high)
    3. Pool of reviews it can choose examples from
    4. Instruction to generate something better in JSON

    Returns the full prompt string.
    """
    history_text = format_history(history)

    # Give optimizer a sample of reviews to pick examples from
    # Show 20 candidate reviews (10 pos + 10 neg) so it has good variety
    pos_candidates = [r for r in candidate_reviews if r["label"] == "positive"][:10]
    neg_candidates = [r for r in candidate_reviews if r["label"] == "negative"][:10]
    all_candidates = pos_candidates + neg_candidates
    random.shuffle(all_candidates)

    candidate_text = ""
    for i, r in enumerate(all_candidates, 1):
        candidate_text += f"  {i}. [{r['label']}] {r['text'][:100]}\n"

    prompt = f"""You are an expert prompt engineer optimizing a sentiment classifier for movie reviews.

TASK:
The classifier reads a movie review and must output exactly 'positive' or 'negative'.
Your goal is to find the BEST combination of:
  1. An instruction that tells the LLM how to classify reviews
  2. Three few-shot examples that best demonstrate correct classification

PAST ATTEMPTS (sorted from worst to best score):
{history_text}
The best score so far is {max(e['accuracy'] for e in history):.1%}.

AVAILABLE REVIEWS TO USE AS EXAMPLES:
{candidate_text}
WHAT TO DO:
Study the pattern — what made higher-scoring attempts better than lower ones?
Then generate a NEW instruction and NEW set of 3 examples that will score HIGHER.

Rules:
- The instruction must be clear and specific about the classification task
- Choose examples that cover different writing styles and difficulty levels
- Include at least one example that handles subtle or ambiguous sentiment
- Do NOT reuse an instruction that already appears in the history above
- Examples must come from the available reviews listed above

Respond with ONLY a valid JSON object in this exact format:
{{
  "instruction": "your new instruction text here",
  "examples": [
    {{"text": "exact review text from the list above", "label": "positive or negative"}},
    {{"text": "exact review text from the list above", "label": "positive or negative"}},
    {{"text": "exact review text from the list above", "label": "positive or negative"}}
  ],
  "reasoning": "one sentence explaining why you think this will score higher"
}}

JSON only. No markdown. No explanation outside the JSON."""

    return prompt


# ── Response Parser
# safely parse JSON responses
def parse_optimizer_response(raw: str) -> dict | None:
    """
    Parse the optimizer's JSON response.

    Returns dict with 'instruction' and 'examples' keys, or None on failure.
    """
    # Strip markdown code fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)

        # Validate required fields
        if "instruction" not in parsed:
            print("    [optimizer] Missing 'instruction' in response")
            return None
        if "examples" not in parsed or len(parsed["examples"]) < 1:
            print("    [optimizer] Missing or empty 'examples' in response")
            return None

        # Validate each example has text + label
        for ex in parsed["examples"]:
            if "text" not in ex or "label" not in ex:
                print(f"    [optimizer] Malformed example: {ex}")
                return None
            if ex["label"] not in ("positive", "negative"):
                print(f"    [optimizer] Invalid label: {ex['label']}")
                ex["label"] = "positive" if "pos" in ex["label"].lower() else "negative"

        return {
            "instruction": parsed["instruction"].strip(),
            "examples"   : parsed["examples"][:NUM_EXAMPLES],
            "reasoning"  : parsed.get("reasoning", "")
        }

    except json.JSONDecodeError as e:
        print(f"    [optimizer] JSON parse error: {e}")
        print(f"    [optimizer] Raw response was: {raw[:200]}")
        return None


# ── Main Optimizer
def generate_new_prompt(
    history: list[dict],
    candidate_reviews: list[dict] | None = None,
    verbose: bool = True
) -> dict | None:
    """
    Given the history of past attempts, generate a new (instruction, examples) combo.

    Parameters
    ----------
    history : list of dict
        Each entry must have: 'instruction', 'examples', 'accuracy'
    candidate_reviews : list of dict, optional
        Pool of reviews the optimizer can choose examples from.
        Defaults to the full optimization set.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict with keys:
        instruction : str   — new instruction text
        examples    : list  — new list of example dicts
        reasoning   : str   — optimizer's explanation
    or None if generation failed after all retries.
    """
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
                max_tokens=800,
                temperature=0.8,   # some creativity — we want diverse suggestions
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


# ── Seed Prompt Generator 
def generate_seed_prompts(n: int = 3) -> list[dict]:
    """
    Generate n diverse seed prompts to kick off the OPRO loop.
    The optimizer needs at least a few examples to work from.

    These are handcrafted starting points — deliberately varied in quality
    so the optimizer has a clear gradient to learn from.
    """
    reviews = get_optimization_set()

    # Pick 9 diverse reviews for seeds (3 per seed prompt)
    pos_reviews = [r for r in reviews if r["label"] == "positive"]
    neg_reviews = [r for r in reviews if r["label"] == "negative"]

    seeds = [
        {
            "instruction": "Classify this movie review as positive or negative.",
            "examples": [
                {"text": pos_reviews[0]["text"], "label": "positive"},
                {"text": neg_reviews[0]["text"], "label": "negative"},
                {"text": pos_reviews[1]["text"], "label": "positive"},
            ]
        },
        {
            "instruction": (
                "Read the movie review below carefully. "
                "Determine whether the reviewer liked or disliked the movie. "
                "Reply with 'positive' if they liked it, 'negative' if they disliked it."
            ),
            "examples": [
                {"text": neg_reviews[1]["text"], "label": "negative"},
                {"text": pos_reviews[2]["text"], "label": "positive"},
                {"text": neg_reviews[2]["text"], "label": "negative"},
            ]
        },
        {
            "instruction": "Is this movie review positive or negative? Answer in one word.",
            "examples": [
                {"text": pos_reviews[3]["text"], "label": "positive"},
                {"text": neg_reviews[3]["text"], "label": "negative"},
                {"text": pos_reviews[4]["text"], "label": "positive"},
            ]
        },
    ]

    return seeds[:n]


# ── Main
if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set!")
        print("Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print("Testing optimizer with mock history...\n")

    # Simulate a small history to test the optimizer
    mock_history = [
        {
            "instruction": "Classify this movie review as positive or negative.",
            "examples"   : [
                {"text": "Loved every minute!", "label": "positive"},
                {"text": "Terrible film.",      "label": "negative"},
                {"text": "A masterpiece.",       "label": "positive"},
            ],
            "accuracy": 0.72
        },
        {
            "instruction": "Is this movie review positive or negative?",
            "examples"   : [
                {"text": "Boring and predictable.", "label": "negative"},
                {"text": "Absolutely fantastic!",  "label": "positive"},
                {"text": "Waste of time.",          "label": "negative"},
            ],
            "accuracy": 0.78
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
    else:
        print("optimizer.py failed to generate a prompt — check API key and errors above.")
