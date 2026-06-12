"""
optimizer.py
------------
The Optimizer LLM — looks at history of past attempts and generates
a new (instruction, examples) combination.

Default optimizer: Qwen2.5-7B-Instruct (smart enough to suggest
good prompts; fits comfortably in 25GB VRAM alongside the scorer)
"""

import os
import sys
import json
import time
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from load_sst5 import get_optimization_set, get_labels


# ── Config 
OPTIMIZER_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
NUM_EXAMPLES      = 5
MAX_HISTORY_SHOWN = 8
MAX_NEW_TOKENS    = 1000

LABELS         = get_labels()
LABEL_LIST_STR = ", ".join(f"'{l}'" for l in LABELS)

# Globals loaded once
_model      = None
_tokenizer  = None
_loaded_for = None


# ── Model Loading 
def load_optimizer_model(model_name: str = OPTIMIZER_MODEL):
    """Load the optimizer model into GPU memory."""
    global _model, _tokenizer, _loaded_for

    if _loaded_for == model_name and _model is not None:
        return

    print(f"Loading optimizer model: {model_name}")
    print(f"  7B model — may take 1-2 minutes on first run...")

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )
    _model.eval()
    _loaded_for = model_name

    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"  ✓ Loaded. VRAM in use: {vram_used:.1f} GB")


# ── History Formatter 
def format_history(history: list[dict]) -> str:
    """Format past attempts sorted from worst to best score."""
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


# ── Optimizer Prompt Builder 
def build_optimizer_prompt(history: list[dict], candidate_reviews: list[dict]) -> str:
    """Build the meta-prompt sent to the Optimizer LLM."""
    history_text = format_history(history)

    # Show 25 candidates (5 per class)
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

    best = max(e["accuracy"] for e in history)

    prompt = f"""You are an expert prompt engineer optimizing a sentiment classifier.

TASK:
Build the best classifier that outputs one of: {LABEL_LIST_STR}

PAST ATTEMPTS (sorted worst to best):
{history_text}
Best score so far: {best:.1%}

AVAILABLE EXAMPLE REVIEWS:
{candidate_text}
Generate a NEW instruction and 5 examples (one per class) that will score HIGHER.

Rules:
- Use exact label spelling: {LABEL_LIST_STR}
- Pick examples from the list above
- Cover all 5 classes
- Don't repeat past instructions

Respond with ONLY valid JSON:
{{
  "instruction": "your new instruction",
  "examples": [
    {{"text": "review text", "label": "label"}},
    {{"text": "review text", "label": "label"}},
    {{"text": "review text", "label": "label"}},
    {{"text": "review text", "label": "label"}},
    {{"text": "review text", "label": "label"}}
  ],
  "reasoning": "one sentence why this is better"
}}"""

    return prompt


# ── Response Parser 
def parse_optimizer_response(raw: str) -> dict | None:
    """Parse the optimizer's JSON response."""
    raw = raw.strip()

    # Strip markdown fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if "{" in part:
                raw = part
                break
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Find JSON object boundaries
    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        print(f"    [optimizer] No JSON object found in response")
        return None
    raw = raw[start:end + 1]

    try:
        parsed = json.loads(raw)

        if "instruction" not in parsed or "examples" not in parsed:
            print(f"    [optimizer] Missing required fields")
            return None

        valid_examples = []
        for ex in parsed["examples"]:
            if "text" not in ex or "label" not in ex:
                continue
            label = ex["label"].strip().lower().replace(" ", "_").replace("-", "_")
            if label not in LABELS:
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
                    continue
            ex["label"] = label
            valid_examples.append(ex)

        if len(valid_examples) == 0:
            return None

        return {
            "instruction": parsed["instruction"].strip(),
            "examples"   : valid_examples[:NUM_EXAMPLES],
            "reasoning"  : parsed.get("reasoning", "")
        }

    except json.JSONDecodeError as e:
        print(f"    [optimizer] JSON parse error: {e}")
        print(f"    [optimizer] Raw was: {raw[:200]}")
        return None


# ── Main Optimizer 
def generate_new_prompt(
    history: list[dict],
    candidate_reviews: list[dict] | None = None,
    verbose: bool = True
) -> dict | None:
    """Given past history, generate a new (instruction, examples) combo."""
    if _model is None:
        load_optimizer_model()

    if candidate_reviews is None:
        candidate_reviews = get_optimization_set()

    optimizer_prompt = build_optimizer_prompt(history, candidate_reviews)

    if verbose:
        best_so_far = max(e["accuracy"] for e in history)
        print(f"\n── Generating new prompt ─────────────────────────────")
        print(f"  History size  : {len(history)} attempts")
        print(f"  Best so far   : {best_so_far:.1%}")
        print(f"  Model         : {_loaded_for}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt engineer. "
                "Respond ONLY with valid JSON, no markdown, no extra text."
            )
        },
        {"role": "user", "content": optimizer_prompt}
    ]

    prompt_text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt_text, return_tensors="pt").to(_model.device)

    for attempt in range(3):
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,    # higher = more creative exploration
                top_p=0.95,
                pad_token_id=_tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw        = _tokenizer.decode(new_tokens, skip_special_tokens=True)
        result     = parse_optimizer_response(raw)

        if result is not None:
            if verbose:
                print(f"  New instruction : {result['instruction'][:80]}...")
                print(f"  Reasoning       : {result['reasoning']}")
            return result

        if verbose:
            print(f"    [optimizer] Parse failed, retrying ({attempt+2}/3)...")

    print(f"    [optimizer] Failed after 3 attempts")
    return None


# ── Seed Prompt Generator 
def generate_seed_prompts(n: int = 3) -> list[dict]:
    """Generate n diverse seed prompts to kick off the OPRO loop."""
    reviews = get_optimization_set()

    by_class = {label: [] for label in LABELS}
    for r in reviews:
        if r["label"] in by_class:
            by_class[r["label"]].append(r)

    def pick_one_per_class(offset):
        return [
            {"text": by_class[label][offset % len(by_class[label])]["text"],
             "label": label}
            for label in LABELS if by_class[label]
        ]

    seeds = [
        {
            # Style 1: Minimalist direct
            "instruction": (
                f"Sentiment label for this movie review: {LABEL_LIST_STR}."
            ),
            "examples": pick_one_per_class(0)
        },
        {
            # Style 2: Verbose with class definitions
            "instruction": (
                "Analyze the movie review and assign a sentiment label using this scale:\n"
                "- 'very_negative': harsh criticism, strong disgust, called bad/terrible\n"
                "- 'negative': clearly disliked, complaints outweigh praise\n"
                "- 'neutral': mixed feelings or balanced/factual description\n"
                "- 'positive': clearly liked, praise outweighs complaints\n"
                "- 'very_positive': strong praise, called masterpiece/brilliant\n"
                "Respond with exactly one label."
            ),
            "examples": pick_one_per_class(1)
        },
        {
            # Style 3: Role-playing / expert framing
            "instruction": (
                "You are a professional film critic categorizing reviews. "
                f"Identify the sentiment as one of: {LABEL_LIST_STR}. "
                "Look at emotional language, intensity words (very, somewhat, absolutely), "
                "and overall tone. Be decisive — pick one label, no hedging."
            ),
            "examples": pick_one_per_class(2)
        },
    ]

    return seeds[:n]


# ── Main 
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    print("Testing optimizer for SST-5 (local model)...\n")

    load_optimizer_model()

    mock_history = [
        {
            "instruction": f"Classify this review. Choose from: {LABEL_LIST_STR}.",
            "examples"   : [
                {"text": "Loved every minute!",      "label": "very_positive"},
                {"text": "Pretty good film.",        "label": "positive"},
                {"text": "It was okay.",             "label": "neutral"},
                {"text": "Boring and slow.",         "label": "negative"},
                {"text": "Worst movie of the year.", "label": "very_negative"},
            ],
            "accuracy": 0.45
        },
    ]

    result = generate_new_prompt(mock_history, verbose=True)

    if result:
        print("\n── Generated Prompt ──────────────────────────────────")
        print(f"Instruction : {result['instruction']}")
        for i, ex in enumerate(result["examples"], 1):
            print(f"  {i}. [{ex['label']}] {ex['text'][:80]}")
        print(f"Reasoning   : {result['reasoning']}")