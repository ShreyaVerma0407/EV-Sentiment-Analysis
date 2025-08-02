#before running code run this in terminal
pip install imbalanced-learn
#code
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('filename.csv')

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Function to classify sentiment using VADER
def classify_sentiment(text):
    sentiment_score = sid.polarity_scores(text)
    # VADER returns a dictionary with negative, neutral, positive scores and a compound score
    if sentiment_score['compound'] >= 0.05:
        return 'positive'
    elif sentiment_score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment classification to the 'comment' column
df['sentiment'] = df['comment'].apply(classify_sentiment)

# Encode sentiment labels for resampling (positive: 1, negative: -1, neutral: 0)
df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': -1, 'neutral': 0})

# Check the distribution before resampling
print("Sentiment distribution before resampling:")
print(df['sentiment'].value_counts())

# Vectorizing the 'comment' column to convert text to numerical features (using TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # You can adjust max_features if needed
X = vectorizer.fit_transform(df['comment']).toarray()  # Converting to an array for SMOTE compatibility
y = df['sentiment_label']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the new distribution after SMOTE
print(f"Resampled training set: {pd.Series(y_train_resampled).value_counts()}")

# Plot the sentiment distribution after resampling
resampled_distribution = pd.Series(y_train_resampled).value_counts()
resampled_distribution.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution After Resampling')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Check if the dataset is balanced
balanced = "balanced" if all(count == resampled_distribution.max() for count in resampled_distribution) else "imbalanced"

# Print whether the dataset is balanced or not after resampling
print(f"The dataset is {balanced} after resampling.")
