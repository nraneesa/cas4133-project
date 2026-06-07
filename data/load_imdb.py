"""
load_imdb.py
------------
Downloads the IMDB movie review dataset from HuggingFace and splits it into:
  - optimization set : 100 reviews  (used during OPRO loop)
  - test set         : 500 reviews  (only touched at final evaluation)

Saves both splits as CSV files in the data/ folder.

MODES:
  --mode online   : download from HuggingFace (use this on Vessel)
  --mode offline  : generate synthetic data   (use this to test pipeline locally)
"""

import os
import argparse
import pandas as pd


# ── Config 
OPTIMIZATION_SIZE = 100   # reviews used inside the OPRO loop
TEST_SIZE         = 500   # reviews reserved for final evaluation
RANDOM_SEED       = 42
SAVE_DIR          = os.path.dirname(os.path.abspath(__file__))


# ── Online: load from HuggingFace 
def load_from_huggingface():
    """Load IMDB dataset from CSV fallback."""
    import urllib.request
    
    csv_path = os.path.join(SAVE_DIR, "IMDB-Dataset.csv")
    
    # Download if not already present
    if not os.path.exists(csv_path):
        print("Downloading IMDB dataset from GitHub...")
        url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
        urllib.request.urlretrieve(url, csv_path)
        print("Download complete!")
    else:
        print("Found existing IMDB-Dataset.csv, loading...")

    df = pd.read_csv(csv_path)
    
    # Rename columns to match expected format
    df.columns = ["text", "label"]          # columns are 'review' and 'sentiment'
    df["label"] = df["label"].str.strip()   # clean whitespace
    
    print(f"  Total reviews loaded : {len(df)}")
    print(f"  Labels found        : {df['label'].unique()}")
    
    # Split into train / test pools
    # train 100 reviews, test other 500 reviews
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    split_at     = int(len(df) * 0.8)
    train_df     = df.iloc[:split_at].reset_index(drop=True)
    test_df_full = df.iloc[split_at:].reset_index(drop=True)
    
    print(f"  Train pool : {len(train_df)} reviews")
    print(f"  Test pool  : {len(test_df_full)} reviews")
    
    return train_df, test_df_full


# ── Offline: generate synthetic data
def load_synthetic():
    """
    Generate a small synthetic dataset that mirrors the real IMDB structure.
    Use this to test the pipeline locally before running on Vessel.
    """
    print("Generating synthetic IMDB-style dataset (offline mode)...")

    positive_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        "A masterpiece of cinema. I laughed, I cried, and I left the theater feeling inspired.",
        "One of the best films I have seen in years. The director did an outstanding job.",
        "Incredible performances from the entire cast. The story was touching and memorable.",
        "A beautifully crafted film with stunning visuals and a deeply moving storyline.",
        "I was blown away by this movie. Every scene was perfectly executed.",
        "The chemistry between the leads was electric. A truly unforgettable experience.",
        "This film exceeded all my expectations. Witty, emotional, and thoroughly entertaining.",
        "A cinematic gem. The soundtrack alone is worth the price of admission.",
        "Brilliantly written and directed. This movie deserves every award it gets.",
        "Loved every minute of it. The pacing was perfect and the ending was satisfying.",
        "An emotional rollercoaster in the best possible way. Highly recommended.",
        "The best movie I have seen this year without a doubt. Simply extraordinary.",
        "A refreshing take on the genre. Creative, fun, and genuinely moving.",
        "Superb filmmaking at its finest. I will be watching this again for sure.",
        "Outstanding in every way. The performances were raw and authentic.",
        "A feel-good movie that will stay with you long after the credits roll.",
        "Thoughtful, funny, and heartfelt. This film has something for everyone.",
        "The writing was sharp and clever. I was captivated from the very first scene.",
        "One of those rare movies that gets better every time you watch it.",
    ]

    negative_reviews = [
        "What a waste of time. The plot made no sense and the acting was terrible.",
        "I fell asleep halfway through. Boring, predictable, and completely forgettable.",
        "The worst movie I have seen in a long time. Avoid at all costs.",
        "Painfully slow with zero character development. I want my two hours back.",
        "The script felt like it was written by someone who had never seen a movie before.",
        "Terrible special effects and a storyline that goes nowhere. Very disappointing.",
        "A complete mess from start to finish. No coherent plot whatsoever.",
        "The acting was cringe-worthy and the dialogue was laughably bad.",
        "I cannot believe this got made. A total disaster of a film.",
        "Dull, pointless, and poorly directed. One of the worst films ever made.",
        "Nothing in this movie works. The humor falls flat and the drama feels forced.",
        "So many plot holes it was impossible to take seriously.",
        "A deeply frustrating watch. Characters made no sense and the ending was abysmal.",
        "Complete garbage. I walked out after an hour.",
        "Poorly written, poorly acted, poorly directed. Just poor all around.",
        "The movie had potential but squandered it with lazy storytelling.",
        "Excruciatingly boring. I kept checking my watch every five minutes.",
        "A terrible film that insulted my intelligence at every turn.",
        "Nothing original here. A lazy rehash of much better movies.",
        "Genuinely one of the most unpleasant viewing experiences I have had.",
    ]

    # Build a large enough pool by cycling through examples
    import random
    random.seed(RANDOM_SEED)

    needed = (OPTIMIZATION_SIZE + TEST_SIZE) // 2 + 10  # per class

    def expand(reviews, n):
        result = []
        while len(result) < n:
            result.extend(reviews)
        return result[:n]

    pos_pool = expand(positive_reviews, needed)
    neg_pool = expand(negative_reviews, needed)

    # Add slight variation to duplicates so they are not identical
    pos_texts = [r if i < len(positive_reviews) else r + f" ({i})" for i, r in enumerate(pos_pool)]
    neg_texts = [r if i < len(negative_reviews) else r + f" ({i})" for i, r in enumerate(neg_pool)]

    all_texts  = pos_texts + neg_texts
    all_labels = ["positive"] * len(pos_texts) + ["negative"] * len(neg_texts)

    full_df = pd.DataFrame({"text": all_texts, "label": all_labels})
    full_df = full_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Split into train pool and test pool
    split_at = len(full_df) // 2
    train_df     = full_df.iloc[:split_at].reset_index(drop=True)
    test_df_full = full_df.iloc[split_at:].reset_index(drop=True)

    print(f"  Synthetic train pool : {len(train_df)} reviews")
    print(f"  Synthetic test pool  : {len(test_df_full)} reviews")
    return train_df, test_df_full


# ── Create balanced splits 
def create_splits(train_df, test_df_full):
    """
    Sample balanced splits (50% positive / 50% negative).

    Returns
    -------
    opt_df  : pd.DataFrame  shape (OPTIMIZATION_SIZE, 2)
    test_df : pd.DataFrame  shape (TEST_SIZE, 2)
    """
    half_opt  = OPTIMIZATION_SIZE // 2
    half_test = TEST_SIZE // 2

    # Optimization set — from train pool
    pos_opt = train_df[train_df["label"] == "positive"].sample(
        min(half_opt, len(train_df[train_df["label"] == "positive"])),
        random_state=RANDOM_SEED
    )
    neg_opt = train_df[train_df["label"] == "negative"].sample(
        min(half_opt, len(train_df[train_df["label"] == "negative"])),
        random_state=RANDOM_SEED
    )
    opt_df = pd.concat([pos_opt, neg_opt]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)[["text", "label"]]

    # Test set — from test pool
    pos_test = test_df_full[test_df_full["label"] == "positive"].sample(
        min(half_test, len(test_df_full[test_df_full["label"] == "positive"])),
        random_state=RANDOM_SEED
    )
    neg_test = test_df_full[test_df_full["label"] == "negative"].sample(
        min(half_test, len(test_df_full[test_df_full["label"] == "negative"])),
        random_state=RANDOM_SEED
    )
    test_df = pd.concat([pos_test, neg_test]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)[["text", "label"]]

    return opt_df, test_df


# ── Save 
def save_splits(opt_df, test_df):
    opt_path  = os.path.join(SAVE_DIR, "optimization_set.csv")
    test_path = os.path.join(SAVE_DIR, "test_set.csv")
    opt_df.to_csv(opt_path,  index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\nSaved optimization set ({len(opt_df)} rows)  → {opt_path}")
    print(f"Saved test set         ({len(test_df)} rows) → {test_path}")


# ── Verify 
def verify_splits(opt_df, test_df):
    print("\n── Optimization Set ──────────────────────────────")
    print(f"  Total  : {len(opt_df)} reviews")
    print(f"  Labels : {opt_df['label'].value_counts().to_dict()}")
    print(f"\n  Sample review:")
    print(f"  Label : {opt_df.iloc[0]['label']}")
    print(f"  Text  : {opt_df.iloc[0]['text'][:200]}...")

    print("\n── Test Set ──────────────────────────────────────")
    print(f"  Total  : {len(test_df)} reviews")
    print(f"  Labels : {test_df['label'].value_counts().to_dict()}")
    print(f"\n  Sample review:")
    print(f"  Label : {test_df.iloc[0]['label']}")
    print(f"  Text  : {test_df.iloc[0]['text'][:200]}...")


# ── Helper functions (used by scorer.py, evaluate.py, etc.) 
def get_optimization_set():
    """Return optimization set as list of dicts: [{'text': ..., 'label': ...}]"""
    path = os.path.join(SAVE_DIR, "optimization_set.csv")
    return pd.read_csv(path).to_dict(orient="records")


def get_test_set():
    """Return test set as list of dicts: [{'text': ..., 'label': ...}]"""
    path = os.path.join(SAVE_DIR, "test_set.csv")
    return pd.read_csv(path).to_dict(orient="records")


# ── Main 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default="offline",
        help="online = HuggingFace (use on Vessel) | offline = synthetic (local testing)"
    )
    args = parser.parse_args()

    if args.mode == "online":
        train_df, test_df_full = load_from_huggingface()
    else:
        train_df, test_df_full = load_synthetic()

    opt_df, test_df = create_splits(train_df, test_df_full)
    save_splits(opt_df, test_df)
    verify_splits(opt_df, test_df)
