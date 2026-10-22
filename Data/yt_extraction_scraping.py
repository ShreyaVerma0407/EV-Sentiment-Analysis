
# Install required libraries
!pip install -q youtube-comment-downloader pandas

import re
import time
import pandas as pd
from youtube_comment_downloader import (
    YoutubeCommentDownloader,
    SORT_BY_RECENT
)
from google.colab import files

# =====================================================
# 1. CONFIGURATION
# =====================================================

VIDEO_URLS = [
    "https://www.youtube.com/watch?v=VIDEO_ID_1",
    "https://www.youtube.com/watch?v=VIDEO_ID_2"
]

MAX_COMMENTS_PER_VIDEO = 500

EV_KEYWORDS = [
    "electric vehicle", "electric vehicles",
    "electric car", "electric cars", "ev",
    "tesla", "byd", "tata nexon ev",
    "battery", "charging station", "charging stations",
    "range anxiety", "ev charging", "electric scooter",
    "electric bike", "electric suv", "electric truck",
    "plug-in hybrid", "charging infrastructure",
    "electric mobility", "fast charging", "battery range",
    "charging time", "ev market", "electric vehicle adoption"
]

# =====================================================
# 2. HELPER FUNCTIONS
# =====================================================

def get_video_id(url):
    match = re.search(
        r"(?:v=|youtu\.be/|shorts/|embed/)"
        r"([A-Za-z0-9_-]{11})",
        url
    )
    return match.group(1) if match else None


def is_ev_related(text):
    text = text.lower()

    for keyword in EV_KEYWORDS:
        if keyword == "ev":
            if re.search(r"\bev\b", text):
                return True
        elif keyword in text:
            return True

    return False


# =====================================================
# 3. EXTRACT COMMENTS
# =====================================================

downloader = YoutubeCommentDownloader()
all_comments = []
seen_ids = set()

for url in VIDEO_URLS:
    print(f"\nProcessing: {url}")

    video_id = get_video_id(url)

    if not video_id:
        print("Invalid YouTube URL. Skipping.")
        continue

    relevant_count = 0
    scanned_count = 0

    try:
        comment_stream = downloader.get_comments_from_url(
            url,
            sort_by=SORT_BY_RECENT
        )

        for comment in comment_stream:
            if relevant_count >= MAX_COMMENTS_PER_VIDEO:
                break

            scanned_count += 1
            text = str(comment.get("text", "")).strip()

            if not text or not is_ev_related(text):
                continue

            comment_id = comment.get("cid", "")

            if comment_id and comment_id in seen_ids:
                continue

            if comment_id:
                seen_ids.add(comment_id)

            all_comments.append({
                "video_id": video_id,
                "comment_id": comment_id,
                "comment": text,
                "author": comment.get("author", ""),
                "likes": comment.get("votes", ""),
                "comment_time": comment.get("time", ""),
                "source_url": url
            })

            relevant_count += 1

        print(f"Comments scanned: {scanned_count}")
        print(f"EV-related comments collected: {relevant_count}")

    except Exception as error:
        print(f"Could not process video: {error}")

    time.sleep(2)

# =====================================================
# 4. SAVE DATASET
# =====================================================

columns = [
    "video_id", "comment_id", "comment", "author",
    "likes", "comment_time", "source_url"
]

df = pd.DataFrame(all_comments, columns=columns)

output_file = "youtube_ev_comments.csv"

df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n========== EXTRACTION SUMMARY ==========")
print("Videos provided:", len(VIDEO_URLS))
print("Total EV-related comments:", len(df))
print("Saved file:", output_file)

if not df.empty:
    display(df.head(10))
else:
    print("No relevant comments collected. Check video URLs or comment availability.")

# =====================================================
# 5. DOWNLOAD CSV
# =====================================================

files.download(output_file)
