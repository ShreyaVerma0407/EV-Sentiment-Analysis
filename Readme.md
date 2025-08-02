# 🧠 Multilabel Text Classification on Social Media Comments  
**Using BERT, RoBERTa, BiLSTM, and a Hybrid Model**

This repository presents four deep learning models designed for multilabel text classification tasks on **YouTube** and **Reddit** comments. These models support:

- 🟢 **Sentiment classification**: Positive, Neutral, Negative  
- 😄 **Emotion detection**: Joy, Sadness, Anger, etc.  
- 🤨 **Sarcasm detection**: Binary sarcastic/not-sarcastic  
- 🌍 **Green tag prediction**: Binary classification for environmentally relevant content  

---

## 📁 Project Structure

```
models/
├── bertmodel.py            # BERT-based classification model
├── Bilstm_model.py         # BiLSTM-based classification model
├── robertamodel.py         # RoBERTa-based classification model
├── hybrid_model.py         # Hybrid (Transformer + BiLSTM) model

data/
├── ev_ytcomments.csv       # YouTube dataset
├── reddit_dataset.csv      # Reddit dataset (for evaluation)
├── Resampling.py
├── ev_dataset_enrichment.py
├── reddit_comments_extraction.py
├── yt_comments_extraction.py
```

---

## 📊 Datasets

### 📍 `data/ev_ytcomments.csv` — YouTube Dataset  
Used for training and in-domain testing in all models.

| Column      | Type    | Description |
|-------------|---------|-------------|
| `comment`   | string  | Text data (YouTube comment) |
| `sentiment` | string  | One of {positive, neutral, negative} |
| `emotion`   | string  | e.g., joy, anger, fear, etc. |
| `sarcasm`   | string  | yes / no |
| `green_tag` | int     | 1 (green topic) / 0 (not) |

---

### 📍 `data/reddit_dataset.csv` — Reddit Dataset  
Used for evaluating cross-domain generalization (more informal, noisy language).

---

## 🎯 Label Tasks

Each model is designed to support **any one of the following label types** at a time:

- `sentiment` (multiclass)
- `emotion` (multiclass)
- `sarcasm` (binary)
- `green_tag` (binary)

To switch tasks, modify:
```python
y = df['sentiment']  # Replace with 'emotion', 'sarcasm', or 'green_tag'
```

---

## 🧠 Model Descriptions

### 1️⃣ BERT Model (`bertmodel.py`)
- **Architecture**: `bert-base-uncased` → Dropout → Linear
- **Tokenizer**: Hugging Face BERT tokenizer
- **Input**: Tokenized comments (max length: 128)
- ✅ Good for: Sentiment
- ⚠️ Limitation: Slight drop in sarcasm/emotion classification

---

### 2️⃣ RoBERTa Model (`robertamodel.py`)
- **Architecture**: `roberta-base` → Dropout → Linear
- **Tokenizer**: Hugging Face RoBERTa tokenizer
- **Input**: Tokenized comments (max length: 256)
- ✅ Good for: General-purpose text tasks
- ⚠️ Limitation: Needs more context for sarcasm

---

### 3️⃣ BiLSTM Model (`Bilstm_model.py`)
- **Architecture**: Word embeddings → 2-layer BiLSTM → FC
- **Tokenizer**: Space-tokenized vocab
- **Input**: Padded word sequences (max length: 256)
- ✅ Good for: Sequence modeling
- ⚠️ Limitation: Weak contextual understanding

---

### 4️⃣ 🔀 Hybrid Model (`hybrid_model.py`)
- **Architecture**:
  - RoBERTa/BERT for context
  - BiLSTM for sequential features
  - Output fusion → FC → Softmax
- ✅ Good for: Sarcasm, emotion, cross-domain performance

---

## 🚀 Running the Models

### 1. Install Dependencies
```bash
pip install torch transformers pandas scikit-learn
```

### 2. Prepare Data
Place your CSV files in the `data/` directory:
```bash
data/ev_ytcomments.csv
data/reddit_dataset.csv
```

### 3. Run a Model
```bash
python bertmodel.py        # Run BERT
python robertamodel.py     # Run RoBERTa
python Bilstm_model.py     # Run BiLSTM
python hybrid_model.py     # Run Hybrid model
```

> ✅ By default, the models train on `sentiment`. Change `y = df['sentiment']` to another label to train for `emotion`, `sarcasm`, or `green_tag`.

---

## 💡 Motivation for Label Expansion

Originally, the dataset supported only **sentiment**. We added more labels for real-world application:

### 🔹 Emotion Label
- Captures fine-grained states (e.g., joy vs. sadness)
- Useful for mental health monitoring & public sentiment analysis

### 🔹 Sarcasm Label
- Sarcastic content flips meaning — useful for improving model robustness

### 🔹 Green Tag Label
- Identifies environment-related content
- Useful for sustainability and eco-focused filtering

---

## ✅ Summary

This repository enables developers and researchers to:
- Build and compare transformer and non-transformer-based models
- Perform multilabel classification on social media text
- Apply models across domains (YouTube vs Reddit)
- Understand emotions, sarcasm, and green-tag relevance in online discussions

---

## 👩‍💻 Author

- Khushi  and shreya

---

#
