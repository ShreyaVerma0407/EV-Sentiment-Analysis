import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('filename.csv')

# Step 1: Remove rows with missing values in the 'comment' column (optional based on your dataset)
df = df.dropna(subset=['comment'])

# Step 2: Preprocess the 'comment' column
def preprocess_text(text):
    # Convert to lowercase to maintain uniformity
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove Emojis (non-ASCII characters)
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Remove mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)

    # Remove hashtags (e.g., #EVs)
    text = re.sub(r'#\w+', '', text)

    # Remove special characters and digits, leaving only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords (words that don't add much meaning)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Return the cleaned text (meaningful content)
    return text

# Apply the preprocessing function to the 'comment' column
df['comment'] = df['comment'].apply(preprocess_text)

# Step 3: Save the preprocessed dataset back to the same file
df.to_csv('filename.csv', index=False)

# Step 4: Print the number of entries left after preprocessing
print(f"Number of entries left after preprocessing: {len(df)}")