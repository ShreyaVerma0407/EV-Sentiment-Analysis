import torch
import torch.nn as nn
from transformers import RobertaModel
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # [batch, seq, 1]
        context = torch.sum(weights * lstm_out, dim=1)       # [batch, hidden*2]
        return context

class RobertaBiLSTMHybridModel(nn.Module):
    def __init__(self, hidden_dim=128, num_labels=3):
        super(RobertaBiLSTMHybridModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, structured_features):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, 768]

        lstm_output, _ = self.lstm(sequence_output)  # [batch, seq_len, hidden*2]
        attn_output = self.attention(lstm_output)    # [batch, hidden*2]

        combined = torch.cat((attn_output, structured_features), dim=1)  # [batch, hidden*2 + 3]
        logits = self.classifier(combined)

        return logits
class EVCommentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.texts = df['comment'].tolist()
        self.sentiment_scores = df['sentiment_score'].values
        self.sarcasm = df['sarcasm'].values
        self.green_tag = df['green_tag'].values
        self.labels = df['label'].values
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        structured = torch.tensor([self.sentiment_scores[idx], self.sarcasm[idx], self.green_tag[idx]], dtype=torch.float)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'structured_features': structured,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        features = batch['structured_features'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, mask, features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc


def eval_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            features = batch['structured_features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, mask, features)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc
from sklearn.model_selection import train_test_split

# Load and preprocess your data
# Get the folder where the script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make full path to the CSV file
csv_path = os.path.join(BASE_DIR, 'Data', 'Datasets', 'ev_redditcomments')

# Read the CSV file using dynamic path
df = pd.read_csv(csv_path)

label_map = {"positive": 0, "neutral": 1, "negative": 2}
df['label'] = df['sentiment'].map(label_map)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Dataset & DataLoader
train_dataset = EVCommentDataset(train_df, tokenizer)
val_dataset = EVCommentDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model
model = RobertaBiLSTMHybridModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
best_acc = 0
patience = 3
counter = 0

for epoch in range(20):  # max 20 epochs
    print(f"\nEpoch {epoch+1}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss, val_acc = eval_model(model, val_loader, loss_fn, device)

    print(f"✅ Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
    print(f"✅ Val Loss:   {val_loss:.4f} | Accuracy: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("💾 Saved best model")
    else:
        counter += 1
        if counter >= patience:
            print("🛑 Early stopping")
            break
