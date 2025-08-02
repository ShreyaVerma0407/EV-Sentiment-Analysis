from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["positive", "neutral", "negative"],
            yticklabels=["positive", "neutral", "negative"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import numpy as np
import re

# --------------------- LOAD & PREPROCESS ---------------------
df = pd.read_csv("ev_with_sentiment_emotion_green_sarcasm.csv")

df['comment'] = df['comment'].astype(str)
df['clean_comment'] = df['comment'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
df['tokens'] = df['clean_comment'].apply(lambda x: x.split())
df['comment_length'] = df['comment'].apply(len)
label_map = {"positive": 0, "neutral": 1, "negative": 2}
df['label'] = df['sentiment'].map(label_map)

# Fake predictions for confusion matrix
pred_labels = np.random.choice([0, 1, 2], size=len(df))
true_labels = df['label'].values

# --------------------- 1. Confusion Matrix ---------------------
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["positive", "neutral", "negative"],
            yticklabels=["positive", "neutral", "negative"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --------------------- 2. Boxplot of Sentiment Score by Emotion ---------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='emotion', y='sentiment_score', palette='Set2')
plt.title("Sentiment Score Distribution by Emotion")
plt.show()

# --------------------- 3. Top 20 Frequent Words ---------------------
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['comment'])
word_freq = np.asarray(X.sum(axis=0)).flatten()
words = vectorizer.get_feature_names_out()
freq_df = pd.DataFrame({'word': words, 'count': word_freq}).sort_values(by='count', ascending=False).head(20)
plt.figure(figsize=(8, 5))
sns.barplot(data=freq_df, x='count', y='word', palette='cubehelix')
plt.title("Top 20 Frequent Words")
plt.show()

# --------------------- 4. Green vs General Top Words ---------------------
green_words = ' '.join(df[df['green_tag'] == 1]['clean_comment'])
general_words = ' '.join(df[df['green_tag'] == 0]['clean_comment'])

green_vector = CountVectorizer(stop_words='english').fit([green_words])
green_counts = green_vector.transform([green_words]).toarray().flatten()
green_freq = pd.DataFrame({'word': green_vector.get_feature_names_out(), 'count': green_counts}).sort_values(by='count', ascending=False).head(10)

general_vector = CountVectorizer(stop_words='english').fit([general_words])
general_counts = general_vector.transform([general_words]).toarray().flatten()
general_freq = pd.DataFrame({'word': general_vector.get_feature_names_out(), 'count': general_counts}).sort_values(by='count', ascending=False).head(10)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=green_freq, y='word', x='count', ax=axes[0], palette='Greens')
axes[0].set_title("Top Green Post Keywords")
sns.barplot(data=general_freq, y='word', x='count', ax=axes[1], palette='Reds')
axes[1].set_title("Top General Post Keywords")
plt.tight_layout()
plt.show()

# --------------------- 5. Density Plot of Sentiment Scores ---------------------
plt.figure(figsize=(8, 5))
sns.kdeplot(df['sentiment_score'], fill=True)
plt.title("Density Distribution of Sentiment Scores")
plt.xlabel("Sentiment Score")
plt.show()

# --------------------- 6. Word Clouds ---------------------
def generate_wordcloud(text, title):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

generate_wordcloud(' '.join(df['clean_comment']), "All Words Word Cloud")
generate_wordcloud(' '.join(df[df['sentiment'] == 'positive']['clean_comment']), "Positive Words Word Cloud")
generate_wordcloud(' '.join(df[df['sentiment'] == 'negative']['clean_comment']), "Negative Words Word Cloud")
generate_wordcloud(' '.join(df[df['sentiment'] == 'neutral']['clean_comment']), "Neutral Words Word Cloud")

# --------------------- 7. Sentiment Score: Green vs General ---------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x='green_tag', y='sentiment_score', data=df, palette='coolwarm')
plt.title("Sentiment Score: Green (1) vs General (0)")
plt.xticks([0, 1], ['General', 'Green'])
plt.show()

# --------------------- 8. Emotion Distribution ---------------------
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='emotion', order=df['emotion'].value_counts().index, palette='pastel')
plt.title("Emotion Distribution (All Data)")
plt.xticks(rotation=45)
plt.show()

# --------------------- 9. Sentiment by Sarcasm ---------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='sentiment', hue='sarcasm', palette='cool')
plt.title("Sentiment Distribution by Sarcasm Label")
plt.xlabel("Sentiment")
plt.legend(title='Sarcasm')
plt.show()

# --------------------- 10. Cleaning Impact on Word Count ---------------------
df['clean_word_count'] = df['clean_comment'].apply(lambda x: len(x.split()))
plt.figure(figsize=(7, 4))
sns.boxplot(data=pd.melt(df[['comment_length', 'clean_word_count']]), x='variable', y='value', palette='Set3')
plt.title("Impact of Cleaning on Word Count")
plt.xlabel("Step")
plt.ylabel("Word Count")
plt.show()

# --------------------- 11. Top Frequent Words After Cleaning ---------------------
flat_words = [word for tokens in df['tokens'] for word in tokens]
clean_word_freq = pd.Series(flat_words).value_counts().head(20)
plt.figure(figsize=(8, 5))
sns.barplot(x=clean_word_freq.values, y=clean_word_freq.index, palette='Blues_r')
plt.title("Top 20 Frequent Words After Cleaning")
plt.show()
# Number of comments by sarcasm label
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='sarcasm', palette='flare')
plt.title("Number of Comments by Sarcasm Label")
plt.xticks([0, 1], ['Non-Sarcastic', 'Sarcastic'])
plt.ylabel("Comment Count")
plt.show()
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
df['token_length'] = df['comment'].apply(lambda x: len(tokenizer.tokenize(x)))

plt.figure(figsize=(8, 5))
sns.histplot(df['token_length'], bins=40, kde=True, color='mediumslateblue')
plt.title("Token Length Distribution of EV Comments")
plt.xlabel("Number of Tokens")
plt.ylabel("Comment Count")
plt.axvline(128, color='red', linestyle='--', label='128 tokens (truncate threshold)')
plt.legend()
plt.show()

df['truncated'] = df['token_length'].apply(lambda x: 1 if x > 128 else 0)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='truncated', palette='coolwarm')
plt.title("Truncated vs Non-Truncated Comments (Max=128 Tokens)")
plt.xticks([0, 1], ['Not Truncated', 'Truncated'])
plt.ylabel("Count")
plt.show()
# ✅ Compare word frequency in sarcastic vs non-sarcastic comments
def get_top_words(text_list, top_n=10):
    all_words = ' '.join(text_list).split()
    return pd.Series(all_words).value_counts().head(top_n)

sarcastic_words = get_top_words(df[df['sarcasm'] == 1]['clean_comment'])
nonsarcastic_words = get_top_words(df[df['sarcasm'] == 0]['clean_comment'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(x=sarcastic_words.values, y=sarcastic_words.index, ax=axes[0], palette='Reds')
axes[0].set_title("Top Words in Sarcastic Comments")

sns.barplot(x=nonsarcastic_words.values, y=nonsarcastic_words.index, ax=axes[1], palette='Greens')
axes[1].set_title("Top Words in Non-Sarcastic Comments")
plt.tight_layout()
plt.show()
# ✅ Define aspect keywords
aspect_keywords = {
    'battery': ['battery', 'charge', 'charging', 'range'],
    'price': ['cost', 'expensive', 'cheap', 'affordable', 'price'],
    'performance': ['speed', 'power', 'acceleration', 'performance'],
    'design': ['design', 'look', 'style'],
    'support': ['service', 'support', 'maintenance']
}

# ✅ Assign aspect labels to comments
def identify_aspect(comment):
    comment = comment.lower()
    for aspect, keywords in aspect_keywords.items():
        if any(word in comment for word in keywords):
            return aspect
    return 'other'

df['aspect'] = df['comment'].apply(identify_aspect)

# ✅ Boxplot of sentiment score per aspect
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[df['aspect'] != 'other'], x='aspect', y='sentiment_score', palette='viridis')
plt.title("Aspect-Based Sentiment Scores")
plt.xlabel("Aspect")
plt.ylabel("Sentiment Score")
plt.show()
# ✅ Setup
model.eval()
text = df.iloc[0]['comment']
true_label = df.iloc[0]['label']

# ✅ Tokenize input
tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
input_ids = tokens['input_ids'].to(device)
attention_mask = tokens['attention_mask'].to(device)

# ✅ Structured features
structured_input = torch.tensor([[df.iloc[0]['sentiment_score'], df.iloc[0]['sarcasm'], df.iloc[0]['green_tag']]], dtype=torch.float).to(device)

# ✅ Forward pass to get hidden states
with torch.no_grad():
    roberta_outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = roberta_outputs.last_hidden_state[0]  # shape: [seq_len, 768]
    token_norms = torch.norm(hidden_states, dim=1).cpu().numpy()  # [seq_len]

# ✅ Normalize attention scores
token_norms = (token_norms - token_norms.min()) / (token_norms.max() - token_norms.min())

# ✅ Decode tokens for x-axis
tokens_decoded = tokenizer.convert_ids_to_tokens(input_ids[0])

# ✅ Plot heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 2))
sns.heatmap([token_norms], xticklabels=tokens_decoded, cmap='coolwarm', cbar=True)
plt.title(f"Token Attention (RoBERTa hidden norm) — True: {true_label}")
plt.xticks(rotation=90)
plt.yticks([])
plt.tight_layout()
plt.show()
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Tokenize and process as before
text = df.iloc[0]['comment']
true_label = df.iloc[0]['label']

tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
input_ids = tokens['input_ids'].to(device)
attention_mask = tokens['attention_mask'].to(device)

structured_input = torch.tensor([[df.iloc[0]['sentiment_score'], df.iloc[0]['sarcasm'], df.iloc[0]['green_tag']]], dtype=torch.float).to(device)

# Forward to get last hidden states
with torch.no_grad():
    roberta_outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = roberta_outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
    token_importance = torch.norm(hidden_states, dim=1).cpu().numpy()

# Convert token IDs to readable tokens
tokens_decoded = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

# Filter: remove [PAD], [CLS], [SEP]
filtered_tokens = []
filtered_scores = []

for token, score in zip(tokens_decoded, token_importance):
    if token not in ['[PAD]', '[CLS]', '[SEP]']:
        filtered_tokens.append(token)
        filtered_scores.append(score)

# Normalize
filtered_scores = np.array(filtered_scores)
filtered_scores = (filtered_scores - filtered_scores.min()) / (filtered_scores.max() - filtered_scores.min())

# Plot cleaned heatmap
plt.figure(figsize=(min(len(filtered_tokens), 14), 2.5))
sns.heatmap([filtered_scores], xticklabels=filtered_tokens, cmap='coolwarm', cbar=True)
plt.title(f"📌 Cleaned Token Attention (True: {true_label})")
plt.xticks(rotation=90, fontsize=9)
plt.yticks([])
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 6))
co_matrix = pd.crosstab(df['emotion'], df['sentiment'])
sns.heatmap(co_matrix, annot=True, cmap='YlGnBu')
plt.title("Emotion vs Sentiment Co-occurrence")
plt.show()
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x='emotion', y='sentiment_score', palette='Spectral')
plt.title("Average Sentiment Score per Emotion")
plt.xticks(rotation=45)
plt.show()
import seaborn as sns
sns.pairplot(df[['sentiment_score', 'sarcasm', 'green_tag']], diag_kind='kde', corner=True)
plt.suptitle("Pairplot: Sentiment, Sarcasm & Green Tag", y=1.02)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

colors = df['label'].map({0: 'green', 1: 'blue', 2: 'red'})

ax.scatter(df['sentiment_score'], df['sarcasm'], df['green_tag'], c=colors)
ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Sarcasm')
ax.set_zlabel('Green Tag')
ax.set_title("3D Feature Space (color = sentiment)")
plt.show()

df['word_count'] = df['comment'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='sentiment', y='word_count', palette='Accent')
plt.title("Comment Length by Sentiment")
plt.show()

from wordcloud import WordCloud

# Green posts
green_text = ' '.join(df[df['green_tag'] == 1]['clean_comment'])
green_wc = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(green_text)

# General posts
general_text = ' '.join(df[df['green_tag'] == 0]['clean_comment'])
general_wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(general_text)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(green_wc, interpolation='bilinear')
axes[0].set_title("Green Posts Word Cloud")
axes[0].axis("off")

axes[1].imshow(general_wc, interpolation='bilinear')
axes[1].set_title("General Posts Word Cloud")
axes[1].axis("off")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 4))
year_counts = df['year'].value_counts().sort_index()
sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o')
plt.title("📈 Year-wise Growth in EV Comments")
plt.xlabel("Year")
plt.ylabel("Number of Comments")
plt.grid(True)
plt.tight_layout()
plt.show()

monthly_counts = df['year_month'].value_counts().sort_index()

plt.figure(figsize=(12, 5))
sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, marker='o')
plt.xticks(rotation=45)
plt.title("📆 Monthly Trend of EV Comments")
plt.xlabel("Year-Month")
plt.ylabel("Comment Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(df['hour'], bins=24, kde=True, color='skyblue')
plt.title("🕒 Comment Activity by Hour of Day")
plt.xlabel("Hour (0–23)")
plt.ylabel("Number of Comments")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()

heatmap_data = pd.crosstab(df['year'], df['emotion'])

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='magma')
plt.title("🔥 Yearly Emotion Heatmap (2020–2025)")
plt.xlabel("Emotion")
plt.ylabel("Year")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='year', hue='emotion', palette='Set2')
plt.title("🎭 Emotion Distribution by Year (2020–2025)")
plt.xlabel("Year")
plt.ylabel("Comment Count")
plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='weekday', order=weekday_order, palette='pastel')
plt.title("📅 Comment Distribution by Weekday")
plt.xlabel("Day of Week")
plt.ylabel("Number of Comments")
plt.tight_layout()
plt.show()

df['date_only'] = df['comment_time'].dt.date
daily_sentiment = df.groupby('date_only')['sentiment_score'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 4))
daily_sentiment.head(15).plot(kind='bar', color='mediumseagreen')
plt.title("🔥 Top 15 Days with Highest Avg Sentiment Score")
plt.ylabel("Average Sentiment Score")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

daily_sentiment.tail(15).plot(kind='bar', color='crimson')
plt.title("💔 Top 15 Most Negative Days")
plt.ylabel("Average Sentiment Score")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

