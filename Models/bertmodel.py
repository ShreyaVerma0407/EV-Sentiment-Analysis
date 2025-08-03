import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, accuracy_score
import os

# Get the folder where the script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make full path to the CSV file
csv_path = os.path.join(BASE_DIR, 'Data', 'Datasets', 'ev_ytcomments.csv')

# Read the CSV file using dynamic path
df = pd.read_csv(csv_path)

X = df['comment'].astype(str).tolist()        # ✅ comment text
y = df['sentiment']                           # ✅ sentiment label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Converts 'positive', 'neutral', etc. to 0, 1, 2
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# Step 4: Tokenize All Texts Upfront (Fast!)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128  # use 128 for speed (256 if needed)

def encode(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')

print("🔁 Pre-tokenizing...")
train_encodings = encode(X_train)
val_encodings = encode(X_val)
test_encodings = encode(X_test)
# Step 5: Fast Dataset
class FastDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = FastDataset(train_encodings, y_train)
val_dataset = FastDataset(val_encodings, y_val)
test_dataset = FastDataset(test_encodings, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)
# Step 6: Model
class SimpleBERTModel(nn.Module):
    def __init__(self, num_labels):
        super(SimpleBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(self.dropout(cls_output))
# Step 7: Training & Evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleBERTModel(num_labels=len(label_encoder.classes_)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=5)

def train(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Accuracy = {acc:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        print(f"Validation Accuracy = {val_acc:.4f}")

        scheduler.step()

    evaluate(model, test_loader)

def evaluate(model, test_loader):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            preds += outputs.argmax(dim=1).cpu().tolist()
            labels_all += labels.cpu().tolist()

print("🚀 Starting fast training...")
train(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs=10)
def test_model_and_report(model, dataloader, device, label_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Test Accuracy: {acc:.4f}\n")

    # Classification Report
    print("📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=label_names, digits=4)
          if label_names else classification_report(all_labels, all_preds, digits=4))
label_names = ["Negative", "Neutral", "Positive"]

test_model_and_report(model, test_loader, device, label_names)
