import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

# Get the folder where the script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make full path to the CSV file
csv_path = os.path.join(BASE_DIR, 'Data', 'Datasets', 'ev_ytcomments.csv')

# Read the CSV file using dynamic path
df = pd.read_csv(csv_path)

# Preprocessing: Handle missing values and convert all text to strings
df['comment'] = df['comment'].fillna('')
df['comment'] = df['comment'].astype(str)

# Tokenizer for RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_len = 256

# Tokenization function
def encode_text(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

# Encode the text and labels
X = df['comment'].tolist()
y = df['sentiment'].values

# Encode sentiments
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split into train, val, test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Dataset class
class EVCarDataset(Dataset):
    def __init__(self, texts, labels):  # ✅ Fixed
        self.texts = texts
        self.labels = labels
        self.encodings = encode_text(texts)

    def __len__(self):  # ✅ Fixed
        return len(self.texts)

    def __getitem__(self, idx):  # ✅ Fixed
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Create dataset objects
train_dataset = EVCarDataset(X_train, y_train)
val_dataset = EVCarDataset(X_val, y_val)
test_dataset = EVCarDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Simple RoBERTa Model
class SimpleRoBERTaModel(nn.Module):
    def __init__(self, num_labels=3):  # ✅ Fixed
        super(SimpleRoBERTaModel, self).__init__()  # ✅ Fixed

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        hidden_states = roberta_output.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        output = self.fc(hidden_states[:, 0, :])
        return output

# Model setup
model = SimpleRoBERTaModel(num_labels=len(label_encoder.classes_))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# Training function
def train(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=10):
    best_val_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_train_predictions = 0
        total_train_predictions = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_train_predictions += (preds == labels).sum().item()
            total_train_predictions += labels.size(0)

        train_accuracy = correct_train_predictions / total_train_predictions
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        correct_val_predictions = 0
        total_val_predictions = 0
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_val_predictions += (preds == labels).sum().item()
            total_val_predictions += labels.size(0)

        val_accuracy = correct_val_predictions / total_val_predictions
        print(f"Validation Loss: {val_loss / len(val_dataloader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

        scheduler.step()

    evaluate_model(model, test_dataloader)

# Evaluation function
def evaluate_model(model, test_dataloader):
    model.eval()
    correct_test_predictions = 0
    total_test_predictions = 0
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            correct_test_predictions += (preds == labels).sum().item()
            total_test_predictions += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = correct_test_predictions / total_test_predictions
    print(f"Test Loss: {test_loss / len(test_dataloader):.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

# Run training
train(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=10)
