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

# Load and process data
df = pd.read_csv('new_ev_dataset_with_emotion.csv')

# Preprocessing: Handle missing values and convert all text to strings
df['comment'] = df['comment'].fillna('')  # Replace missing comments with empty strings
df['comment'] = df['comment'].astype(str)  # Ensure all comments are strings

# Tokenizer for RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
max_len = 256  # Maximum length of tokenized input


# Tokenization function
def encode_text(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")


# Encode the text and labels
X = df['comment'].tolist()  # Use 'comment' for text data
y = df['sentiment'].values  # Use 'sentiment' for labels

# Encode sentiments (if they are not numerical)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test-validation split (80% train, 10% validation, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Dataset class for handling data
class EVCarDataset(Dataset):
    def _init_(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.encodings = encode_text(texts)

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Creating datasets
train_dataset = EVCarDataset(X_train, y_train)
val_dataset = EVCarDataset(X_val, y_val)
test_dataset = EVCarDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)


# Simplified RoBERTa Model (No BiLSTM or Attention)
class SimpleRoBERTaModel(nn.Module):
    def _init_(self, num_labels=3):
        super(SimpleRoBERTaModel, self)._init_()

        # RoBERTa model
        self.roberta = RobertaModel.from_pretrained('roberta-base')  # Use roberta-base

        # Fully connected layer for output
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_labels)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids, attention_mask=attention_mask)
        hidden_states = roberta_output.last_hidden_state

        # Apply dropout for regularization
        hidden_states = self.dropout(hidden_states)

        # Get the final output (using the [CLS] token)
        output = self.fc(hidden_states[:, 0, :])  # Using the first token (CLS token) for classification
        return output


# Initialize the model
model = SimpleRoBERTaModel(num_labels=len(label_encoder.classes_))

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Learning Rate Scheduler (CosineAnnealingLR)
scheduler = CosineAnnealingLR(optimizer, T_max=10)


# Train the model
def train(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=10):
    best_val_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct_train_predictions = 0
        total_train_predictions = 0

        # Training loop
        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate accuracy
            _, preds = torch.max(outputs, dim=1)
            correct_train_predictions += (preds == labels).sum().item()
            total_train_predictions += labels.size(0)

        train_accuracy = correct_train_predictions / total_train_predictions
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation loop
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

        # Save model after each epoch if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

        # Step the scheduler
        scheduler.step()

    # Evaluate on the test set after training
    evaluate_model(model, test_dataloader)


# Evaluate the model on the test set
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


# Start training (run all epochs)
train(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=10)