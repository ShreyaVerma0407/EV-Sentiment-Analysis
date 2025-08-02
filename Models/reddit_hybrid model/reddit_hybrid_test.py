import torch
from torch.utils.data import Dataset

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
        inputs = self.tokenizer(text, truncation=True, padding='max_length',
                                max_length=self.max_len, return_tensors="pt")
        structured = torch.tensor(
            [self.sentiment_scores[idx], self.sarcasm[idx], self.green_tag[idx]],
            dtype=torch.float
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'structured_features': structured,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
val_dataset = EVCommentDataset(val_df, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=16)
model = RobertaBiLSTMHybridModel()
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
from sklearn.metrics import accuracy_score, classification_report
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        features = batch['structured_features'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, mask, features)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("✅ Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=["positive", "neutral", "negative"]))
