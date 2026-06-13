import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load new CSV
df_new = pd.read_csv("data/new_tweets.csv")

# Load existing label encoder
label_encoder = torch.load("models/label_encoder.pt")

# Encode new labels
df_new['Label_encoded'] = label_encoder.transform(df_new['Label'])

# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained("models/distilbert_model")
model = AutoModelForSequenceClassification.from_pretrained("models/distilbert_model")

# Prepare Dataset
dataset_new = Dataset.from_pandas(df_new[['Comment', 'Label_encoded']])
dataset_new = dataset_new.map(lambda x: tokenizer(x['Comment'], padding=True, truncation=True, max_length=128), batched=True)
dataset_new.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label_encoded'])

# Fine-tune on new data
training_args = TrainingArguments(
    output_dir="models/",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_strategy="no",
    logging_dir="logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_new,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("models/distilbert_model")
tokenizer.save_pretrained("models/distilbert_model")
