!pip install praw  # Install PRAW in Colab

import praw
import pandas as pd
import time
from datetime import datetime

# Authenticate with Reddit API using your credentials
reddit = praw.Reddit(
    client_id="your_client_id",  # Replace with your client_id
    client_secret="your_client_secret",  # Replace with your client_secret
    user_agent="your_user_agent"  # Replace with your user agent
)

# Subreddit to scrape
subreddit = reddit.subreddit("electricvehicles")

# Function to fetch comments in batches (test with 100 comments)
def fetch_reddit_comments(subreddit, target_count=100, batch_size=100):
    comments = []
    after = None

    while len(comments) < target_count:
        params = {'subreddit': subreddit, 'size': batch_size}
        if after:
            params['after'] = after  # Pagination to get next batch of comments

        # Fetch comments from Reddit API
        submissions = subreddit.top(limit=batch_size)  # Getting top submissions (adjust this as necessary)
        for submission in submissions:
            submission.comments.replace_more(limit=0)  # Get all comments without replacing
            for comment in submission.comments.list():
                comment_time = datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')  # Convert UTC timestamp to readable format
                comments.append({
                    'title': submission.title,  # Extracting title of the post
                    'comment': comment.body,  # Extracting comment body
                    'score': comment.score,  # Extracting score (upvotes - downvotes)
                    'comment_time': comment_time  # Extracting the time of the comment
                })

            # Get the 'after' timestamp to fetch the next batch of comments
            after = submission.created_utc

            # Print a message after each batch
            if len(comments) % 100 == 0:
                print(f"{len(comments)} comments fetched so far...")

            # Sleep for 1 second to prevent rate limiting
            time.sleep(1)

        # If the length of comments is still less than the target count, keep looping
        if len(comments) >= target_count:
            break

    return comments


# Fetch comments (set to 100 comments)
comments = fetch_reddit_comments(subreddit, target_count=100)

# Convert to DataFrame and save as CSV
if comments:
    df = pd.DataFrame(comments)
    df.to_csv('filename.csv', index=False)
    print(f"Data saved to 'filename.csv'. Total comments fetched: {len(comments)}")
else:
    print("No comments were fetched.")
