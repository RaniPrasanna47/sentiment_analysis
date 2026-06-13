from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from collections import Counter
from preprocess_sentiment_analysis import preprocess_dataframe, wordcloud_to_base64
import seaborn as sns
import numpy as np

app = FastAPI(title="Sentiment Analysis API")

API_KEY = "mysecret1234"  # Replace with your secret key

# ---------------- API Key Verification ----------------
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

# ---------------- Helpers ----------------
def ngrams_from_text(text: str, n: int):
    tokens = text.split()
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def compute_clause_sentiment(df: pd.DataFrame, category_col: str):
    """Returns a dict {Clause: {Positive, Neutral, Negative}}"""
    if "Clause" not in df.columns or df["Clause"].dropna().empty:
        return {}
    clause_data = {}
    for clause, group in df.groupby("Clause"):
        counts = group[category_col].value_counts(normalize=True) * 100
        clause_data[clause] = {
            "Positive": round(counts.get("Positive", 0), 2),
            "Neutral": round(counts.get("Neutral", 0), 2),
            "Negative": round(counts.get("Negative", 0), 2)
        }
    return clause_data

def generate_clause_sentiment_chart(clause_data: dict):
    if not clause_data:
        return None
    df_chart = pd.DataFrame(clause_data).T[["Positive", "Neutral", "Negative"]]
    n_clauses = len(df_chart)
    index = np.arange(n_clauses)
    bar_width = 0.25

    plt.figure(figsize=(10, max(4, n_clauses * 0.5)))
    plt.barh(index - bar_width, df_chart["Positive"], height=bar_width, color="green", label="Positive")
    plt.barh(index, df_chart["Neutral"], height=bar_width, color="gray", label="Neutral")
    plt.barh(index + bar_width, df_chart["Negative"], height=bar_width, color="red", label="Negative")
    plt.yticks(index, df_chart.index)
    plt.xlabel("Percentage (%)")
    plt.ylabel("Clause")
    plt.title("Clause Sentiment Distribution")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_wordcount_distribution(df: pd.DataFrame, category_col: str):
    """Return list of dicts [{category: ..., word_count: ...}, ...]"""
    if "clean_comment" not in df.columns:
        return []
    df_wc = df.copy()
    df_wc["word_count"] = df_wc["clean_comment"].str.split().str.len()
    # Map categories to numeric for plotting (optional)
    if df_wc[category_col].dtype != str:
        df_wc[category_col] = df_wc[category_col].astype(str)
    return df_wc[[category_col, "word_count"]].rename(columns={category_col: "category"}).to_dict(orient="records")

# ---------------- Analyze Endpoint ----------------
@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    # Preprocess dataframe (clean_comment column, etc.)
    df_processed = preprocess_dataframe(df)

    # Determine category column
    category_col = next((c for c in ["Label", "label", "Category", "category"] if c in df_processed.columns), None)
    if not category_col:
        raise HTTPException(status_code=400, detail="No sentiment/category column found in CSV")

    # All text for wordclouds and N-grams
    all_text = " ".join(df_processed["clean_comment"].astype(str).tolist())

    analysis = {}
    analysis["sentiment_counts"] = df_processed[category_col].value_counts().to_dict()
    analysis["top_words"] = Counter(all_text.split()).most_common(50)
    analysis["top_bigrams"] = Counter(ngrams_from_text(all_text, 2)).most_common(25)
    analysis["top_trigrams"] = Counter(ngrams_from_text(all_text, 3)).most_common(25)
    analysis["wordcloud_base64"] = wordcloud_to_base64(all_text)

    analysis["total_comments"] = len(df_processed)
    analysis["unique_clause"] = int(df_processed["Clause"].nunique()) if "Clause" in df_processed.columns else 0
    analysis["avg_word_count"] = float(df_processed["clean_comment"].str.split().str.len().mean())
    analysis["sample_processed"] = df_processed[["clean_comment", category_col, "Clause"]].head(10).to_dict(orient="records")

    # Clause sentiment
    clause_data = compute_clause_sentiment(df_processed, category_col)
    analysis["clause_sentiment"] = clause_data
    analysis["clause_sentiment_chart_base64"] = generate_clause_sentiment_chart(clause_data)

    # Word count distribution by sentiment
    analysis["word_count_data"] = generate_wordcount_distribution(df_processed, category_col)

    # Clean comments for word clouds by sentiment
    if "clean_comment" in df_processed.columns:
        sentiment_map = {category_col: {"Positive": 'Positive', "Neutral": 'Neutral', "Negative": 'Negetive'}}
        df_processed["category_numeric"] = df_processed[category_col].map(lambda x: sentiment_map.get(category_col, {}).get(x, 0))
        analysis["clean_comments"] = df_processed[["clean_comment", "category_numeric"]].rename(columns={"category_numeric":"category"}).to_dict(orient="records")

    return JSONResponse(content=analysis)

@app.get("/")
def root():
    return {"status": "ok", "message": "Sentiment Analysis API is running."}
