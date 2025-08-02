import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("new_ev_dataset_with_emotion.csv")
df['comment'] = df['comment'].astype(str)
df['label'] = df['sentiment'].map({'positive': 0, 'neutral': 1, 'negative': 2})

# Normalize structured features
scaler = StandardScaler()
df[['sentiment_score', 'sarcasm', 'green_tag']] = scaler.fit_transform(df[['sentiment_score', 'sarcasm', 'green_tag']])

# Split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class EVCommentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['comment'].tolist()
        self.scores = df['sentiment_score'].values
        self.sarcasm = df['sarcasm'].values
        self.green = df['green_tag'].values
        self.labels = df['label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.texts[idx], truncation=True, padding='max_length',
                                max_length=self.max_len, return_tensors="pt")
        features = torch.tensor([self.scores[idx], self.sarcasm[idx], self.green[idx]], dtype=torch.float)
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'structured_features': features,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

train_loader = DataLoader(EVCommentDataset(train_df, tokenizer), batch_size=16, shuffle=True)
test_loader = DataLoader(EVCommentDataset(test_df, tokenizer), batch_size=16)

# Attention and Model class
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

class RobertaHybrid(nn.Module):
    def __init__(self, hidden_dim=128, num_labels=3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, structured_features):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(output.last_hidden_state)
        attn_out = self.attn(lstm_out)
        x = torch.cat((attn_out, structured_features), dim=1)
        return self.classifier(x)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        CE = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-CE)
        return ((1 - pt) ** self.gamma * CE).mean()

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaHybrid().to(device)

# Freeze first 3 layers of RoBERTa
for param in model.roberta.embeddings.parameters():
    param.requires_grad = False
for layer in model.roberta.encoder.layer[:3]:
    for param in layer.parameters():
        param.requires_grad = False

# Class weights and loss
class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = FocalLoss(alpha=class_weights)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
epochs = 25
total_steps = len(train_loader) * epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Train + Eval
def train_epoch(model, loader):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        features = batch['structured_features'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, mask, features)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc

def eval_model(model, loader):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
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
    return total_loss / len(loader), acc

# Resume training if needed
best_acc = 0
patience = 5
counter = 0
checkpoint_path = "best_model_focal.pth"

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    val_loss, val_acc = eval_model(model, test_loader)
    best_acc = val_acc
    print(f"✅ Loaded model from {checkpoint_path} with best accuracy: {best_acc:.4f}")
else:
    print("🟡 No saved checkpoint found — training from scratch")

# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = eval_model(model, test_loader)
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Model Saved with Accuracy: {val_acc:.4f}")
    else:
        counter += 1
        print(f"🔁 No improvement. Early stop counter: {counter}/{patience}")
        if counter >= patience:
            print("🛑 Early stopping")
            break
