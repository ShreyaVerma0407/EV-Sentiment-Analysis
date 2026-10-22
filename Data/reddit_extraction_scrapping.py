
!pip install -q praw pandas

import praw
import pandas as pd
from datetime import datetime, timezone

# 1. Connect to Reddit API
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="EV-Sentiment-Analysis/1.0 by YOUR_REDDIT_USERNAME"
)

# 2. Select the subreddit
subreddit = reddit.subreddit("electricvehicles")

# 3. Extract comments
def fetch_reddit_comments(subreddit, target_count=100):
    comments_data = []
    seen_ids = set()

    # Fetch top posts from the past year
    for submission in subreddit.top(time_filter="year", limit=100):

        # Skip posts that are not relevant to EVs
        post_text = (
            submission.title + " " + (submission.selftext or "")
        ).lower()

        ev_keywords = [
            "electric vehicle", "electric car", "ev ",
            "tesla", "evs", "battery", "charging",
            "electric truck", "electric suv"
        ]

        if not any(keyword in post_text for keyword in ev_keywords):
            continue

        # Retrieve the available comment tree
        submission.comments.replace_more(limit=0)

        for comment in submission.comments.list():
            if not isinstance(comment, praw.models.Comment):
                continue

            if comment.id in seen_ids:
                continue

            if comment.body in ("[deleted]", "[removed]"):
                continue

            seen_ids.add(comment.id)

            comments_data.append({
                "post_id": submission.id,
                "comment_id": comment.id,
                "subreddit": str(subreddit),
                "title": submission.title,
                "comment": comment.body,
                "score": comment.score,
                "comment_time": datetime.fromtimestamp(
                    comment.created_utc, tz=timezone.utc
                ).isoformat(),
                "post_url": submission.url
            })

            if len(comments_data) >= target_count:
                break

        print(f"Collected {len(comments_data)} comments")

        if len(comments_data) >= target_count:
            break

    return comments_data


# 4. Run extraction
comments = fetch_reddit_comments(subreddit, target_count=100)

# 5. Save to CSV
if comments:
    df = pd.DataFrame(comments)
    df.to_csv("reddit_ev_comments.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} comments to reddit_ev_comments.csv")
    display(df.head())
else:
    print("No comments collected. Check API access or post availability.")
