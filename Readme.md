# 🧠 Multilabel Text Classification on Social Media Comments  
**Using BERT, RoBERTa, BiLSTM, and a Hybrid Model**

This repository presents four deep learning models designed for multilabel text classification tasks on **YouTube** and **Reddit** comments. These models support:

- 🟢 **Sentiment classification**: Positive, Neutral, Negative  
- 😄 **Emotion detection**: Joy, Sadness, Anger, etc.  
- 🤨 **Sarcasm detection**: Binary sarcastic/not-sarcastic  
- 🌍 **Green tag prediction**: Binary classification for environmentally relevant content  

---




## 📁 Project Structure

models
├── bertmodel.py # BERT-based classification model
├── Bilstm_model.py # BiLSTM-based classification model
├── robertamodel.py # RoBERTa-based classification model
├── hybrid_model.py # Hybrid (Transformer + BiLSTM) model
└── data/datsets
├── ev_ytcomments.csv # YouTube dataset
└── reddit_dataset.csv # Reddit dataset (for evaluation)
├── Resampling.py
├── ev_dataset_enrichment.py
├── reddit_comments_extraction.py
├── yt_comments_extraction.py

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

To switch tasks, modify the line:
```python
y = df['sentiment']  # Replace with 'emotion', 'sarcasm', or 'green_tag'
🧠 Model Descriptions
1️⃣ BERT Model (bertmodel.py)
Architecture: bert-base-uncased → Dropout → Linear

Tokenizer: Hugging Face BERT tokenizer

Input: Tokenized comments (max length: 128)

Strength: Good contextual representation for sentiment

Limitations: Slight drop in sarcasm/emotion classification

Evaluation: Classification report + accuracy

2️⃣ RoBERTa Model (robertamodel.py)
Architecture: roberta-base → Dropout → Linear

Tokenizer: Hugging Face RoBERTa tokenizer

Input: Tokenized comments (max length: 256)

Strength: Strong general-purpose transformer

Limitations: Needs more context for sarcasm

Evaluation: Detailed metrics via classification_report

3️⃣ BiLSTM Model (Bilstm_model.py)
Architecture: Custom word-level embedding → 2-layer BiLSTM → FC

Tokenizer: Basic space-tokenized vocab

Input: Padded word sequences (max length: 256)

Strength: Strong in capturing sequence dependencies

Limitations: Weak contextual understanding

Evaluation: Accuracy, precision, recall, F1

4️⃣ 🔀 Hybrid Model (hybrid_model.py)
Architecture:

RoBERTa/BERT for global context

BiLSTM for sequential nuance

Output fusion → FC → Softmax

Input: Transformer-encoded + tokenized

Strength: Best results on sarcasm, emotion, cross-domain

Evaluation: Tested on YouTube and Reddit
🚀 Running the Models
1. Install Dependencies
bash
Copy
Edit
pip install torch transformers pandas scikit-learn
2. Prepare Data
Place your CSV files in the data/ directory:

bash
Copy
Edit
data/ev_ytcomments.csv
data/reddit_dataset.csv
3. Run a Model
bash
Copy
Edit
python bertmodel.py       # Run BERT
python robertamodel.py    # Run RoBERTa
python Bilstm_model.py    # Run BiLSTM
python hybrid_model.py    # Run Hybrid model
✅ By default, the models train on sentiment. Change y = df['sentiment'] to another label if needed.

 How and Why New Labels Were Added
Originally, the dataset supported only sentiment. We expanded it for real-world applications:

🔹 Emotion Label
Captures fine-grained mental states (e.g., joy vs. sadness)

Useful for mental health monitoring and public opinion analysis

🔹 Sarcasm Label
Sarcastic content often flips the sentiment meaning

Improves model robustness in informal online platforms

🔹 Green Tag Label
Filters for environment-related discussions

Useful for sustainability analysis or targeted filtering
