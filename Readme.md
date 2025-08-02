# 🧠 Model Overview

This document describes the architecture and role of each model used in the project.

---

## 🔹 BERT Classifier (`bertmodel.py`)
- Transformer: `bert-base-uncased`
- Task: Sentiment classification (default), adaptable for emotion/sarcasm/green_tag
- Layers: BERT encoder → Dropout → Fully connected
- Tokenization: Hugging Face's BERT tokenizer
- Input: Max 128 tokens
- Dataset: YouTube comments (`data/ev_ytcomments.csv`)

---

## 🔹 BiLSTM Classifier (`Bilstm_model.py`)
- Architecture: 2-layer BiLSTM + Embedding + Dropout + FC
- Tokenization: Custom space-based
- Embeddings: Learned from scratch
- Input: Word sequences (max length 256)
- Dataset: YouTube comments

---

## 🔹 RoBERTa Classifier (`robertamodel.py`)
- Transformer: `roberta-base`
- Task: Multilabel support for sentiment/emotion/sarcasm
- Layers: RoBERTa encoder → Dropout → FC (on [CLS] token)
- Tokenization: Hugging Face RoBERTa tokenizer
- Dataset: YouTube comments (extended with emotions)

---

## 🧪 Evaluation
- Each model evaluates on the YouTube dataset.
- Optional: Reddit dataset (`data/reddit_dataset.csv`) used for cross-domain testing.

# 📊 Data Description

This document outlines the dataset files, their structure, and labels used in this project.

---

## 📁 Location
All datasets are stored in the `data/` folder.

---

### 1. `ev_ytcomments.csv` (Primary dataset)
- **Source**: YouTube Comments
- **Used For**: Training, validation, and testing
- **Columns**:
  - `comment` → Text content
  - `sentiment` → {positive, neutral, negative}
  - `emotion` → {joy, anger, sadness, fear, etc.}
  - `sarcasm` → {yes, no}
  - `green_tag` → {1, 0}

---

### 2. `reddit_dataset.csv` (Evaluation only)
- **Source**: Reddit posts/comments
- **Used For**: Testing model generalization
- **Label Format**: Same as YouTube dataset

---

## 🧪 Label Notes
- `sentiment`, `emotion`, `sarcasm`, and `green_tag` can all be selected individually.
- Only one label is used at a time per training session.
# 🚀 Usage Guide

Instructions for training and evaluating models.

---

## 🧰 Prerequisites
Install dependencies:

```bash
pip install torch transformers scikit-learn pandas
