import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR

# Load and process data
df = pd.read_csv('/kaggle/input/yt-balanced/new_ev_dataset_with_emotion.csv')

# Check for missing values and handle them
df['comment'] = df['comment'].fillna('')  # Replace NaN values with empty string

# Ensure that all comments are strings
df['comment'] = df['comment'].astype(str)

# Preprocessing: Tokenization and padding for BiLSTM
max_len = 256  # Maximum length of tokenized input

# Use the correct column name for text data
X = df['comment'].tolist()  # Use 'comment' column for text data
y = df['sentiment'].values  # Sentiment labels

# Encode sentiments if they are not numerical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-test-validation split (80% train, 10% validation, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Tokenization (assuming it's simple space-based tokenization for BiLSTM)
def tokenize_text(texts):
    return [text.split() for text in texts]  # Tokenize based on spaces (or use any other method)


# Dataset class for handling data
class EVCarDataset(Dataset):
    def _init_(self, texts, labels, max_len):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenized_texts = tokenize_text(texts)
        self.vocab = self.build_vocab(self.tokenized_texts)

    def build_vocab(self, tokenized_texts):
        vocab = {}
        index = 0
        for text in tokenized_texts:
            for word in text:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
        return vocab

    def encode_text(self, text):
        return [self.vocab.get(word, 0) for word in text][:self.max_len]  # Word index for each text

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, idx):
        text = self.tokenized_texts[idx]
        encoded_text = self.encode_text(text)
        # Padding the encoded text
        encoded_text = encoded_text + [0] * (self.max_len - len(encoded_text))

        return {
            'input_ids': torch.tensor(encoded_text, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


train_dataset = EVCarDataset(X_train, y_train, max_len)
val_dataset = EVCarDataset(X_val, y_val, max_len)
test_dataset = EVCarDataset(X_test, y_test, max_len)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)


# Full BiLSTM Model
class FullBiLSTMModel(nn.Module):
    def _init_(self, vocab_size, embed_dim=100, hidden_dim=256, num_labels=3):
        super(FullBiLSTMModel, self)._init_()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # BiLSTM Layer (two layers of BiLSTM)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim * 2, num_labels)  # BiLSTM is bidirectional (hidden_dim * 2)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids):
        # Embedding layer
        embedded = self.embedding(input_ids)

        # BiLSTM Layer
        lstm_out, _ = self.lstm(embedded)

        # Apply dropout for regularization
        lstm_out = self.dropout(lstm_out)

        # Use the last hidden state for classification
        output = self.fc(lstm_out[:, -1, :])  # Using the last token's hidden state for classification
        return output


# Initialize the model
model = FullBiLSTMModel(vocab_size=len(train_dataset.vocab), num_labels=len(label_encoder.classes_))

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids)
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
            labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(input_ids)

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

        # Check if model achieved target accuracy
        if val_accuracy > 0.90:
            print("Achieved target accuracy!")
            break

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
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
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


# Start training
train(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=10)