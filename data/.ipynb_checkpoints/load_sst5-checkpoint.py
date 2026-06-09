"""
load_sst5.py
------------
Loads the SST-5 (Stanford Sentiment Treebank, 5-class) dataset.

Five sentiment classes:
  - very_negative
  - negative
  - neutral
  - positive
  - very_positive

This is a HARDER task than IMDB binary sentiment, which gives OPRO
much more room to demonstrate improvement.

Saves splits as CSV files in the data/ folder.
"""

import os
import argparse
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────
OPTIMIZATION_SIZE = 100   # reviews used inside the OPRO loop
TEST_SIZE         = 500   # reviews reserved for final evaluation
RANDOM_SEED       = 42
SAVE_DIR          = os.path.dirname(os.path.abspath(__file__))

# Label mapping for SST-5 (integer label → text label)
LABEL_MAP = {
    0: "very_negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very_positive",
}

# List of all valid labels (used by scorer for validation)
LABELS = list(LABEL_MAP.values())


# ── Online: load from HuggingFace ─────────────────────────────────────────────
def load_from_huggingface():
    """
    Load SST-5 from HuggingFace. The SetFit/sst5 mirror is the most reliable.
    """
    from datasets import load_dataset
    print("Downloading SST-5 dataset from HuggingFace (SetFit/sst5)...")
    dataset = load_dataset("SetFit/sst5")

    print(f"  Train split : {len(dataset['train'])} samples")
    print(f"  Test split  : {len(dataset['test'])} samples")

    train_df     = pd.DataFrame(dataset["train"])
    test_df_full = pd.DataFrame(dataset["test"])

    # SetFit/sst5 already has 'label_text' column with strings like
    # 'very negative', 'negative', 'neutral', 'positive', 'very positive'
    # Normalize to underscore format for consistency
    def normalize(s):
        return s.strip().lower().replace(" ", "_")

    train_df["label"]     = train_df["label_text"].apply(normalize)
    test_df_full["label"] = test_df_full["label_text"].apply(normalize)

    # Standardize the text column name
    train_df     = train_df.rename(columns={"text": "text"})[["text", "label"]]
    test_df_full = test_df_full.rename(columns={"text": "text"})[["text", "label"]]

    print(f"  Unique labels : {sorted(train_df['label'].unique())}")
    return train_df, test_df_full


# ── Create balanced splits ────────────────────────────────────────────────────
def create_splits(train_df, test_df_full):
    """
    Sample roughly balanced splits across the 5 classes.

    Returns
    -------
    opt_df  : pd.DataFrame
    test_df : pd.DataFrame
    """
    per_class_opt  = OPTIMIZATION_SIZE // 5    # 20 per class
    per_class_test = TEST_SIZE         // 5    # 100 per class

    def sample_balanced(df, per_class):
        """Sample `per_class` from each label, then shuffle."""
        chunks = []
        for label in LABELS:
            available = df[df["label"] == label]
            n         = min(per_class, len(available))
            if n > 0:
                chunks.append(available.sample(n=n, random_state=RANDOM_SEED))
        out = pd.concat(chunks).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        return out[["text", "label"]]

    opt_df  = sample_balanced(train_df,     per_class_opt)
    test_df = sample_balanced(test_df_full, per_class_test)

    return opt_df, test_df


# ── Save ──────────────────────────────────────────────────────────────────────
def save_splits(opt_df, test_df):
    opt_path  = os.path.join(SAVE_DIR, "optimization_set.csv")
    test_path = os.path.join(SAVE_DIR, "test_set.csv")
    opt_df.to_csv(opt_path,  index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\nSaved optimization set ({len(opt_df)} rows)  → {opt_path}")
    print(f"Saved test set         ({len(test_df)} rows) → {test_path}")


# ── Verify ────────────────────────────────────────────────────────────────────
def verify_splits(opt_df, test_df):
    print("\n── Optimization Set ──────────────────────────────")
    print(f"  Total  : {len(opt_df)} samples")
    print(f"  Labels : {opt_df['label'].value_counts().to_dict()}")
    print(f"\n  Sample row:")
    print(f"  Label : {opt_df.iloc[0]['label']}")
    print(f"  Text  : {opt_df.iloc[0]['text']}")

    print("\n── Test Set ──────────────────────────────────────")
    print(f"  Total  : {len(test_df)} samples")
    print(f"  Labels : {test_df['label'].value_counts().to_dict()}")
    print(f"\n  Sample row:")
    print(f"  Label : {test_df.iloc[0]['label']}")
    print(f"  Text  : {test_df.iloc[0]['text']}")


# ── Helper functions (used by scorer.py, evaluate.py, etc.) ───────────────────
def get_optimization_set():
    """Return optimization set as list of dicts: [{'text': ..., 'label': ...}]"""
    path = os.path.join(SAVE_DIR, "optimization_set.csv")
    return pd.read_csv(path).to_dict(orient="records")


def get_test_set():
    """Return test set as list of dicts: [{'text': ..., 'label': ...}]"""
    path = os.path.join(SAVE_DIR, "test_set.csv")
    return pd.read_csv(path).to_dict(orient="records")


def get_labels():
    """Return list of valid labels."""
    return LABELS.copy()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["online"],
        default="online",
        help="online = HuggingFace (works on Vessel)"
    )
    args = parser.parse_args()

    train_df, test_df_full = load_from_huggingface()
    opt_df, test_df         = create_splits(train_df, test_df_full)
    save_splits(opt_df, test_df)
    verify_splits(opt_df, test_df)

    print(f"\nSST-5 dataset ready!")
    print(f"Task difficulty: HARD (5 classes)")
    print(f"Expected gpt-3.5-turbo baseline: 45-55%")
