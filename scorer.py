"""
scorer.py
---------
The Scorer LLM — evaluates how good a (instruction, examples) combo is
by testing it against the optimization set and returning an accuracy score.

Default scorer: Qwen2.5-1.5B-Instruct

Usage:
    from scorer import score_prompt, load_scorer_model

    load_scorer_model()  # Load once at startup
    result = score_prompt(instruction, examples)
"""

import os
import sys
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
from load_sst5 import get_optimization_set, get_labels


# ── Config 
SCORER_MODEL   = "microsoft/Phi-3-mini-4k-instruct"  # default scorer model
BATCH_SIZE     = 100
MAX_NEW_TOKENS = 10

# Valid labels for SST-5
LABELS         = get_labels()
LABEL_LIST_STR = ", ".join(f"'{l}'" for l in LABELS)

# Globals loaded once on first call
_model      = None
_tokenizer  = None
_loaded_for = None


# ── Model Loading 
def load_scorer_model(model_name: str = SCORER_MODEL):
    """
    Load the scorer model into GPU memory. Call once at startup.

    Reuses the model across calls so we don't reload every time.
    """
    global _model, _tokenizer, _loaded_for

    if _loaded_for == model_name and _model is not None:
        return  # already loaded

    print(f"Loading scorer model: {model_name}")
    print(f"  This takes ~30 seconds on first run...")

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    _loaded_for = model_name

    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"  ✓ Loaded. VRAM in use: {vram_used:.1f} GB")


# ── Prompt Builder 
def build_prompt(instruction: str, examples: list[dict], review_text: str) -> str:
    """Build the full prompt for a single review classification."""
    lines = [instruction.strip(), ""]

    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"Review: {ex['text'].strip()}")
        lines.append(f"Label: {ex['label'].strip()}")
        lines.append("")

    lines.append("Now classify this review.")
    lines.append(f"Review: {review_text.strip()}")
    lines.append(f"Label (respond with ONLY one of: {LABEL_LIST_STR}):")

    return "\n".join(lines)


# ── Response Parser 
def parse_label(raw: str) -> str:
    """Parse LLM response into one of the 5 valid labels."""
    raw = raw.strip().lower()
    cleaned = raw.replace(" ", "_").replace("-", "_")

    # Check very_* labels first (so 'positive' doesn't match 'very_positive')
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


# ── Single Review Classifier 
def classify_review(instruction: str, examples: list[dict], review_text: str) -> str:
    """Classify a single review using the local model."""
    global _model, _tokenizer

    if _model is None:
        load_scorer_model()

    user_content = build_prompt(instruction, examples, review_text)

    # Use chat template for instruct models
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a fine-grained sentiment classifier. "
                f"Respond with ONLY one label from: {LABEL_LIST_STR}. No explanation."
            )
        },
        {"role": "user", "content": user_content}
    ]

    prompt_text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt_text, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    return parse_label(raw)


# ── Main Scorer 
def score_prompt(
    instruction: str,
    examples: list[dict],
    reviews: list[dict] | None = None,
    verbose: bool = True
) -> dict:
    """Score a (instruction, examples) combo on the optimization set."""
    if _model is None:
        load_scorer_model()

    if reviews is None:
        reviews = get_optimization_set()

    reviews = reviews[:BATCH_SIZE]

    if verbose:
        print(f"\n── Scoring prompt ────────────────────────────────────")
        print(f"  Instruction : {instruction[:80]}{'...' if len(instruction) > 80 else ''}")
        print(f"  Examples    : {len(examples)} provided")
        print(f"  Reviews     : {len(reviews)} to score")
        print(f"  Model       : {_loaded_for}")
        print()

    correct     = 0
    predictions = []
    t_start     = time.time()

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
            elapsed_s   = time.time() - t_start
            rate        = (i + 1) / elapsed_s
            print(f"  [{i+1:>3}/{len(reviews)}] Running acc: {running_acc:.1%}  ({rate:.1f} reviews/sec)")

    accuracy = correct / len(reviews) if reviews else 0.0

    if verbose:
        elapsed = time.time() - t_start
        print(f"\n── Result ────────────────────────────────────────────")
        print(f"  Accuracy : {accuracy:.1%}  ({correct}/{len(reviews)} correct)")
        print(f"  Time     : {elapsed:.1f} seconds")

    return {
        "instruction": instruction,
        "examples"   : examples,
        "accuracy"   : accuracy,
        "correct"    : correct,
        "total"      : len(reviews),
        "predictions": predictions
    }


# ── Baseline Scorer 
def score_baseline_prompts(save_path: str = "results/phi_baseline_scores.json"):
    """Score 3 baseline prompts for the SST-5 task."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    reviews = get_optimization_set()

    # Pick one example per class
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
            "description": "No instruction details, no examples"
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
    print("BASELINE SUMMARY (SST-5, local model)")
    print(f"{'='*55}")
    print(f"Model: {_loaded_for}")
    for r in results:
        print(f"  {r['name']:<30} {r['accuracy']:.1%}")
    print(f"\nSaved to {save_path}")
    print()

    return results


# ── Main 
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        print("This script needs a GPU. Run on Vessel.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Task: SST-5 (5-class sentiment classification)")
    print(f"Labels: {LABELS}")
    print()

    load_scorer_model()
    score_baseline_prompts()