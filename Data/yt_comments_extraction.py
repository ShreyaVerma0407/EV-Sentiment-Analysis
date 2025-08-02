!pip
install
google - api - python - client  # Install Google API Client

import pandas as pd
from googleapiclient.discovery import build
import time
from datetime import datetime

# Authenticate with the YouTube API using your API key
YOUTUBE_API_KEY = "YOUR_API_KEY"  # Replace with your API key

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


# Function to search for videos based on a keyword and fetch comments
def search_and_fetch_youtube_comments(keyword, target_count=100, batch_size=100):
    comments = []
    next_page_token = None

    # Search for videos related to the keyword (e.g., "electric vehicles")
    search_request = youtube.search().list(
        part="snippet",
        q=keyword,  # Search query
        type="video",
        maxResults=batch_size
    )

    search_response = search_request.execute()

    # Loop through the search results (videos)
    for item in search_response['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']

        # Fetch comments for the found video
        video_comments = fetch_video_comments(video_id, video_title, target_count, batch_size)
        comments.extend(video_comments)

        if len(comments) >= target_count:
            break

    return comments


# Function to fetch comments for a specific YouTube video
def fetch_video_comments(video_id, video_title, target_count=100, batch_size=100):
    comments = []
    next_page_token = None

    while len(comments) < target_count:
        # Request to get comments from a specific video
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=batch_size,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comment_time = datetime.strptime(comment['publishedAt'], '%Y-%m-%dT%H:%M:%S.%fZ').strftime(
                '%Y-%m-%d %H:%M:%S')
            comments.append({
                'video_title': video_title,  # Extracting video title
                'comment': comment['textDisplay'],  # Extracting comment body
                'score': comment['likeCount'],  # Extracting like count for the comment
                'comment_time': comment_time  # Extracting the time of the comment
            })

        # Get the next page token if more comments exist
        next_page_token = response.get('nextPageToken')

        # Print progress
        if len(comments) % 100 == 0:
            print(f"{len(comments)} comments fetched so far...")

        # Sleep for 1 second to prevent rate limiting
        time.sleep(1)

        # If comments fetched are greater than target, break the loop
        if len(comments) >= target_count:
            break

    return comments


# Example usage: Fetch comments for "electric vehicles"
keyword = "electric vehicles"  # Search for videos related to "electric vehicles"
comments = search_and_fetch_youtube_comments(keyword, target_count=100)

# Convert to DataFrame and save as CSV
if comments:
    df = pd.DataFrame(comments)
    df.to_csv('yt_comments_ev.csv', index=False)
    print(f"filename.csv'. Total comments fetched: {len(comments)}")
else:
    print("No comments were fetched.")
