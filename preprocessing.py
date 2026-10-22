!pip install -q pandas emoji

import pandas as pd
import re
import html
import emoji
from pathlib import Path

# ==========================================================
# 1. LOAD DATASETS
# ==========================================================

REDDIT_FILE = "reddit_ev_comments.csv"
YOUTUBE_FILE = "youtube_ev_comments.csv"

def load_dataset(file_path, platform):
    path = Path(file_path)

    if not path.exists():
        print(f"{file_path} not found. Skipping this dataset.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "comment" not in df.columns:
        raise ValueError(
            f"{file_path} must contain a 'comment' column."
        )

    df["platform"] = platform
    print(f"{platform}: {len(df)} rows loaded")

    return df


reddit_df = load_dataset(REDDIT_FILE, "Reddit")
youtube_df = load_dataset(YOUTUBE_FILE, "YouTube")

# ==========================================================
# 2. COMBINE DATASETS
# ==========================================================

datasets = [df for df in [reddit_df, youtube_df] if not df.empty]

if not datasets:
    raise ValueError("No datasets loaded. Check the CSV filenames.")

df = pd.concat(datasets, ignore_index=True, sort=False)

print("\nTotal rows before cleaning:", len(df))

# ==========================================================
# 3. REMOVE MISSING AND EMPTY COMMENTS
# ==========================================================

df = df.dropna(subset=["comment"]).copy()

df["comment"] = df["comment"].astype(str).str.strip()

df = df[
    (df["comment"] != "") &
    (~df["comment"].str.lower().isin(
        ["nan", "none", "[deleted]", "[removed]"]
    ))
].copy()

print("After removing empty/deleted comments:", len(df))

# ==========================================================
# 4. REMOVE EXACT DUPLICATES
# ==========================================================

df = df.drop_duplicates(subset=["platform", "comment"])

print("After removing platform-level duplicates:", len(df))

# ==========================================================
# 5. TEXT PREPROCESSING FUNCTION
# ==========================================================

def clean_text(text):
    text = str(text)

    # Decode HTML entities such as &amp; and &#39;
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(
        r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
        " ",
        text
    )

    # Remove emojis and their variation sequences
    text = emoji.replace_emoji(text, replace=" ")

    # Remove Reddit-style user mentions
    text = re.sub(r"(?<!\w)u/\w+", " ", text, flags=re.I)

    # Remove subreddit references, e.g. r/electricvehicles
    text = re.sub(r"(?<!\w)r/\w+", " ", text, flags=re.I)

    # Convert text to lowercase
    text = text.lower()

    # Preserve letters, numbers, apostrophes and basic EV terminology.
    # Replace other punctuation and special characters with spaces.
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # Remove standalone apostrophes and normalize whitespace
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Keep original comments and create a separate cleaned column
df["cleaned_comment"] = df["comment"].apply(clean_text)

# ==========================================================
# 6. REMOVE COMMENTS EMPTY AFTER CLEANING
# ==========================================================

df = df[df["cleaned_comment"].str.strip() != ""].copy()

print("After text cleaning:", len(df))

# ==========================================================
# 7. REMOVE DUPLICATES AFTER NORMALIZATION
# ==========================================================

df = df.drop_duplicates(subset=["cleaned_comment"])

print("After normalized duplicate removal:", len(df))

# ==========================================================
# 8. SAVE CLEANED DATASETS
# ==========================================================

# Combined cleaned dataset
df.to_csv(
    "ev_comments_cleaned_combined.csv",
    index=False,
    encoding="utf-8-sig"
)

# Save each platform separately
df[df["platform"] == "Reddit"].to_csv(
    "reddit_ev_comments_cleaned.csv",
    index=False,
    encoding="utf-8-sig"
)

df[df["platform"] == "YouTube"].to_csv(
    "youtube_ev_comments_cleaned.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# 9. DISPLAY CLEANING SUMMARY
# ==========================================================

print("\n========== PREPROCESSING SUMMARY ==========")
print("Final total comments:", len(df))
print("\nComments by platform:")
print(df["platform"].value_counts())

print("\nSample before and after cleaning:")
display(
    df[["platform", "comment", "cleaned_comment"]].head(10)
)

print("\nSaved files:")
print("1. ev_comments_cleaned_combined.csv")
print("2. reddit_ev_comments_cleaned.csv")
print("3. youtube_ev_comments_cleaned.csv")

