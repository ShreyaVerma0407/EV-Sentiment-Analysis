#follwing are the snippets of code that can be used to generate visualisations

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['positive', 'neutral', 'negative'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
save_plot(plt, "confusion_matrix.png")  # Save the plot using plt
plt.show()

# Sentiment Distribution Plot
fig = plt.figure()
sns.countplot(x=all_preds)
plt.xticks(ticks=[0, 1, 2], labels=['positive', 'neutral', 'negative'])
plt.title("Predicted Label Distribution")
plt.xlabel("Predicted Label")
plt.ylabel("Count")
save_plot(fig, "predicted_label_distribution.png")

# Year-wise Growth in EV Comments
fig = plt.figure(figsize=(7, 4))
df['year'].value_counts().sort_index().plot(marker='o')
plt.title("Year-wise Growth in EV Comments")
plt.xlabel("Year")
plt.ylabel("Number of Comments")
plt.grid(True)
save_plot(fig, "year_wise_growth.png")

# Sentiment Score by Emotion
fig = plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x='emotion', y='sentiment_score', palette='Set2')
plt.title("Sentiment Score by Emotion")
plt.xticks(rotation=45)
save_plot(fig, "sentiment_score_by_emotion.png")

# Top 20 Frequent Words in Cleaned Comments
vectorizer = CountVectorizer(stop_words='english')
word_matrix = vectorizer.fit_transform(df['clean_comment'].dropna())  # Make sure no NaNs
word_freq = np.asarray(word_matrix.sum(axis=0)).flatten()
words = vectorizer.get_feature_names_out()
freq_df = pd.DataFrame({'word': words, 'count': word_freq})
freq_df = freq_df.sort_values(by='count', ascending=False).head(20)

fig = plt.figure(figsize=(10, 6))
sns.barplot(data=freq_df, x='count', y='word', palette='coolwarm')
plt.title("Top 20 Frequent Words in Cleaned Comments")
plt.xlabel("Frequency")
plt.ylabel("Word")
save_plot(fig, "top_20_frequent_words.png")

# Sentiment Score Density Distribution
fig = plt.figure(figsize=(7, 4))
sns.kdeplot(df['sentiment_score'], fill=True)
plt.title("Density Distribution of Sentiment Scores")
plt.xlabel("Sentiment Score")
save_plot(fig, "sentiment_score_density.png")

# Word Cloud for All Comments
def generate_wordcloud(text, title, filename):
    wc = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14)
    save_plot(fig, filename)

generate_wordcloud(' '.join(df['clean_comment'].dropna()), "All Comments Word Cloud", "all_comments_wordcloud.png")
generate_wordcloud(' '.join(df[df['sentiment'] == 'positive']['clean_comment'].dropna()), "Positive Word Cloud", "positive_wordcloud.png")
generate_wordcloud(' '.join(df[df['sentiment'] == 'neutral']['clean_comment'].dropna()), "Neutral Word Cloud", "neutral_wordcloud.png")
generate_wordcloud(' '.join(df[df['sentiment'] == 'negative']['clean_comment'].dropna()), "Negative Word Cloud", "negative_wordcloud.png")

# Sentiment Score Comparison: Green vs General
fig = plt.figure(figsize=(7, 4))
sns.boxplot(x='green_tag', y='sentiment_score', data=df, palette='Set1')
plt.title("Sentiment Score Comparison: Green vs General")
plt.xticks([0, 1], ['General', 'Green'])
save_plot(fig, "sentiment_score_green_vs_general.png")

# Yearly Emotion Heatmap
emotion_heatmap = pd.crosstab(df['year'], df['emotion'])
fig = plt.figure(figsize=(14, 6))
sns.heatmap(emotion_heatmap, cmap='magma', annot=True, fmt='d')
plt.title("Yearly Emotion Heatmap")
save_plot(fig, "yearly_emotion_heatmap.png")

# Emotion Distribution in Latest Year
latest_year = df['year'].dropna().max()
fig = plt.figure(figsize=(7, 4))
sns.countplot(data=df[df['year'] == latest_year], x='emotion', order=df['emotion'].value_counts().index, palette='pastel')
plt.title(f"Emotion Distribution in {int(latest_year)}")
plt.xticks(rotation=45)
save_plot(fig, "emotion_distribution_latest_year.png")

# Sentiment Distribution by Sarcasm
fig = plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='sentiment', hue='sarcasm', palette='coolwarm')
plt.title("Sentiment Distribution by Sarcasm")
save_plot(fig, "sentiment_by_sarcasm.png")

# Top 20 Words After Cleaning
flat_words = [word for tokens in df['tokens'] for word in tokens]
clean_freq = pd.Series(flat_words).value_counts().head(20)

fig = plt.figure(figsize=(8, 5))
sns.barplot(x=clean_freq.values, y=clean_freq.index, palette='viridis')
plt.title("Top 20 Words After Cleaning")
save_plot(fig, "top_20_words_after_cleaning.png")

# Sentiment Score vs Likes
fig = plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sentiment_score', y='likes', hue='sentiment')
plt.title("Sentiment Score vs Likes")
plt.xlabel("Sentiment Score")
plt.ylabel("Likes")
plt.grid(True)
save_plot(fig, "sentiment_score_vs_likes.png")

# Comment Length by Sentiment
fig = plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='sentiment', y='comment_length', palette='Set3')
plt.title("Comment Length by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Comment Length")
save_plot(fig, "comment_length_by_sentiment.png")

# Time of Day vs Number of Comments
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df['hour'] = df['time'].dt.hour
fig = plt.figure(figsize=(10, 4))
sns.histplot(df['hour'].dropna(), bins=24, kde=True)
plt.title("Time of Day vs Number of Comments")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Comments")
plt.xticks(range(0, 24))
plt.grid(True)
save_plot(fig, "time_of_day_vs_comments.png")

# 3D Scatter: Sentiment Score vs Likes vs Comment Length
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['sentiment_score'], df['likes'], df['comment_length'], c=df['sentiment_score'], cmap='coolwarm')
ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Likes')
ax.set_zlabel('Comment Length')
ax.set_title('3D Sentiment Score vs Likes vs Comment Length')
save_plot(fig, "3d_sentiment_vs_likes_vs_length.png")

# Top 15 Emotions in Comments
fig = plt.figure(figsize=(10, 5))
df['emotion'].value_counts().head(15).plot(kind='bar', color='skyblue')
plt.title("Top 15 Emotions in Comments")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(True)
save_plot(fig, "top_15_emotions.png")

# Sentiment Score Distribution by Emotion
fig = plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='emotion', y='sentiment_score', palette='Spectral')
plt.title("Sentiment Score Distribution by Emotion")
plt.xticks(rotation=90)
save_plot(fig, "sentiment_score_by_emotion_distribution.png")

# Correlation Heatmap
fig = plt.figure(figsize=(6, 5))
sns.heatmap(df[['sentiment_score', 'likes', 'comment_length', 'word_count']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
save_plot(fig, "correlation_heatmap.png")

# Average Sentiment Score by Year
avg_sentiment_by_year = df.groupby('year')['sentiment_score'].mean()
fig = plt.figure(figsize=(8, 5))
avg_sentiment_by_year.plot(marker='o', title='Average Sentiment Score by Year')
save_plot(fig, "avg_sentiment_by_year.png")

# Sentiment by Emotion vs Likes
fig = plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='emotion', y='likes', estimator=np.mean)
plt.xticks(rotation=90)
plt.title("Average Likes by Emotion")
save_plot(fig, "avg_likes_by_emotion.png")

# Sarcasm Ratio per Year
sarcasm_ratio = df.groupby('year')['sarcasm'].mean()
fig = plt.figure(figsize=(8, 5))
sarcasm_ratio.plot(kind='bar', title='Sarcasm Ratio per Year')
save_plot(fig, "sarcasm_ratio_per_year.png")

# Sentiment Score Distribution by Sarcasm Label
fig = plt.figure(figsize=(8, 5))
sns.boxplot(x='sarcasm', y='sentiment_score', data=df, palette='coolwarm')
plt.title('Sentiment Score Distribution by Sarcasm Label')
plt.xticks([0, 1], ['Not Sarcastic', 'Sarcastic'])
save_plot(fig, "sentiment_score_by_sarcasm_label.png")

# Sentiment Label Distribution by Sarcasm
fig = plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment', hue='sarcasm', data=df, palette='Set1')
plt.title('Sentiment Label Distribution by Sarcasm')
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.legend(title="Sarcasm", labels=['Not Sarcastic', 'Sarcastic'])
save_plot(fig, "sentiment_label_by_sarcasm.png")
