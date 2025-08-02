import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# ✅ Download VADER lexicon
nltk.download('vader_lexicon')

# ✅ Load new dataset
df = pd.read_csv("/kaggle/input/ytcomments/yt_comments_ev_30k.csv")

# ✅ Clean 'comment' column
df['comment'] = df['comment'].astype(str)

# ✅ Add sentiment score using VADER
sid = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['comment'].apply(lambda x: sid.polarity_scores(x)['compound'])

# ✅ Add green tag (sustainability-related)
green_keywords = [
    'green', 'sustainable', 'eco', 'eco-friendly', 'environment', 'carbon',
    'zero emission', 'electric', 'renewable', 'solar', 'climate'
]

def contains_green_words(text):
    text = str(text).lower()
    return int(any(keyword in text for keyword in green_keywords))

df['green_tag'] = df['comment'].apply(contains_green_words)

# ✅ Add sarcasm tag using rule-based logic
sarcasm_keywords = [
    "yeah right", "totally", "as if", "sure thing", "great job", "love that for us",
    "what a surprise", "nice work", "couldn’t be better", "exactly what I needed",
    "just perfect", "of course", "obviously", "classic", "brilliant", "genius",
    "because that makes sense", "love waiting", "excellent idea"
]
sarcasm_emojis = ["🙄", "😒", "😑", "😏", "🤨"]

def is_sarcastic(text):
    text = str(text).lower()
    keyword_match = any(kw in text for kw in sarcasm_keywords)
    emoji_match = any(emoji in text for emoji in sarcasm_emojis)
    praise_words = ["great", "awesome", "amazing", "fantastic", "perfect"]
    negative_words = ["problem", "broke", "fail", "slow", "annoying", "bug"]
    mixed_sentiment = any(p in text for p in praise_words) and any(n in text for n in negative_words)
    return int(keyword_match or emoji_match or mixed_sentiment)

df['sarcasm'] = df['comment'].apply(is_sarcastic)

# ✅ Encode sentiment into labels
label_map = {"positive": 0, "neutral": 1, "negative": 2}
df['label'] = df['sentiment'].map(label_map)

# ✅ Save enriched dataset (optional)
df.to_csv("new_ev_dataset_enriched.csv", index=False)

# ✅ Preview
df.head()