import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load CSV
df = pd.read_csv("data/twitter_validation1.csv")  # Columns: Comment, Label

# Encode labels
le = LabelEncoder()
df['Label_encoded'] = le.fit_transform(df['Label'])
torch.save(le, "models/label_encoder.pt")

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df[['Comment', 'Label_encoded']])
dataset = dataset.train_test_split(test_size=0.2)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['Comment'], padding=True, truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label_encoded'])

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(le.classes_)
)

# Training
training_args = TrainingArguments(
    output_dir="models/",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=50,
    learning_rate=5e-5,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("models/distilbert_model")
tokenizer.save_pretrained("models/distilbert_model")
