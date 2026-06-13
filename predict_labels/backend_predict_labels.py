from fastapi import FastAPI, UploadFile, File
from fastapi import FastAPI, UploadFile, File, HTTPException,Header
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from preproces_predict_labels import preprocess_text
from captum.attr import IntegratedGradients
import torch.nn.functional as F
import torch 
import io
from fastapi import Depends
import pandas as pd
import uvicorn
import joblib


# ------------------ Load model & tokenizer ------------------
MODEL_PATH = "models/my_model"
LE_PATH = "models/label_encoder.pkl"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
label_encoder =joblib.load(LE_PATH)

# ------------------ FastAPI app ------------------
app = FastAPI(title="DistilBERT Text Classification with Live Update")

API_KEY = "mysecret1234"  # Replace with your secret key

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# ------------------ Request schema ------------------
class TextRequest(BaseModel):
    text: str

# ---------------- Text preprocessing ----------------
def preprocess_text(text):
    # Add your preprocessing steps here
    return text.strip()

# ------------------ Prediction endpoint ------------------

@app.post("/predict")
def predict(request: TextRequest):
    try:
        # Preprocess text first
        text_clean = preprocess_text(request.text)
        labels = ["Negative", "Neutral", "Positive"]

        # Tokenize
        inputs = tokenizer(
            text_clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

        # Decode label
        predicted_class = torch.argmax(scores).item()
        label = labels[predicted_class]
        predicted_score = round(float(scores[predicted_class]), 4)

        return {
            "prediction": label,
            "Confidence": predicted_score
        }
    except Exception as e:
        return {"error": str(e)}


# # ------------------ Live update endpoint ------------------
# @app.post("/update_model")
# async def update_model(file: UploadFile = File(...)):
#     # Read new CSV
#     df_new = pd.read_csv(file.file)
#     if 'Comment' not in df_new.columns or 'Label' not in df_new.columns:
#         return {"error": "CSV must have 'Comment' and 'Label' columns"}

#     # Encode labels using existing label encoder
#     df_new['Label_encoded'] = label_encoder.transform(df_new['Label'])
#     df_new["Comment"] = df_new["Comment"].apply(preprocess_text)
#     # Prepare HuggingFace dataset
#     dataset_new = Dataset.from_pandas(df_new[['Comment', 'Label_encoded']])
#     dataset_new = dataset_new.map(lambda x: tokenizer(x['Comment'], padding=True, truncation=True, max_length=128), batched=True)
#     dataset_new.set_format(type='torch', columns=['input_ids', 'attention_mask', 'Label_encoded'])

#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir="models/",
#         num_train_epochs=1,
#         per_device_train_batch_size=16,
#         save_strategy="no",
#         logging_dir="logs",
#     )

#     # Trainer for fine-tuning
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset_new,
#         tokenizer=tokenizer,
#     )

#     trainer.train()
#     model.save_pretrained(MODEL_PATH)
#     tokenizer.save_pretrained(MODEL_PATH)

#     return {"status": "Model updated with new CSV successfully!"}



# ---------------- Classification logic ----------------
labels = ['Negative', 'Neutral', 'Positive']

def classify_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    scores = torch.nn.functional.softmax(output.logits, dim=1)[0]

    # Get predicted class
    predicted_class = torch.argmax(scores).item()
    predicted_label = labels[predicted_class]
    predicted_score = round(float(scores[predicted_class]),4)

    return predicted_label, predicted_score


# ------------------ CSV classification endpoint -----------------
@app.post("/classify_file")
async def classify_file(
    file: UploadFile = File(...),
    verified: bool = Depends(verify_api_key)  # Dependency will enforce API key
):
    # Get file extension
    filename = file.filename
    ext = os.path.splitext(filename)[-1].lower()
    
    try:
        contents = await file.read()
        
        if ext == ".csv":
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(io.BytesIO(contents))
        elif ext == ".txt":
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')), delimiter="\t")
        else:
            return {"error": f"Unsupported file type: {ext}"}
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    # Ensure Comment column exists
    if "Comment" not in df.columns:
        return {"error": "File must have a 'Comment' column"}

    if "ID" not in df.columns:
        df["ID"] = range(1, len(df) + 1)

    # Preprocess text
    df["Comment"] = df["Comment"].apply(preprocess_text)

    labels = []
    scores = []

    for sentence in df["Comment"]:
        try:
            label, score = classify_text(sentence)
        except Exception:
            label, score = "Error", 0.0
        labels.append(label)
        scores.append(score)

    df["Label"] = labels
    df["Confidence"] = scores
    # df = df[["ID", "Comment", "Label", "Confidence"]]

    # Save CSV
    output_file = os.path.abspath("labeled_data.csv")
    df.to_csv(output_file, index=False)

    return FileResponse(output_file, filename="labeled_data.csv", media_type='text/csv')



# ------------------ Health check ------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------ Run server ------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



