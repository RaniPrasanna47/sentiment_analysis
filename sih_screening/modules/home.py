
import streamlit as st
import requests
import pandas as pd
import numpy as np
import io, os
from st_aggrid import AgGrid
import matplotlib.pyplot as plt
import base64
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter


# ---------------- API URLs ----------------
PREDICT_API = "http://127.0.0.1:8500/predict"
# CLASSIFY_API = "http://127.0.0.1:8000/classify_file"
ANALYZE_API = "http://127.0.0.1:8500/analyze"

DB_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__),"..", "..", "database", "virtual_db.json")
)
# ---------------- Helper Functions ---------------

def create_kde_by_category(df, value_col, category_col, title, figsize=(6, 4)):
        fig, ax = plt.subplots(figsize=figsize)
        categories = df[category_col].unique()
        for cat in categories:
            subset = df[df[category_col] == cat]
            if subset.empty:
                continue
            sns.kdeplot(subset[value_col], label=str(cat), ax=ax, fill=True)
        ax.set_title(title)
        ax.set_xlabel(value_col)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)



def plot_word_cloud(text_series, title):
    if len(text_series) == 0:
        st.warning(f"⚠️ No data for {title}")
        return
    text = " ".join(text_series.dropna().astype(str))
    wc = WordCloud(width=600, height=400, background_color="white", colormap="Set2").generate(text)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.subheader(title)
    st.pyplot(fig)


def plot_top_words_by_category(df, text_col, category_col, n=20, figsize=(10, 6)):
    all_counts = {}
    categories = df[category_col].unique()
    
    # Collect top words for each category
    for cat in categories:
        subset = df[df[category_col] == cat]
        words = " ".join(subset[text_col].dropna().astype(str)).split()
        word_freq = Counter(words)
        all_counts[cat] = dict(word_freq.most_common(n))

    # Convert into DataFrame
    top_words_df = pd.DataFrame(all_counts).fillna(0).astype(int)

    # Define colors for categories
    color_map = {
        "positive": "green",
        "negative": "red",
        "neutral": "#FF8C00"
    }
    # Strip whitespace and convert to lowercase for proper mapping
    colors = [color_map.get(cat.strip().lower(), "red") for cat in top_words_df.columns]

    # Plot stacked bar
    ax = top_words_df.plot(kind="bar", stacked=True, figsize=figsize, color=colors)
    ax.set_title(f"Top {n} Words by Sentiment Category")
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha='right', fontsize=5)  # adjust size
    plt.tight_layout()
    st.pyplot(plt.gcf())

def plot_sentiment_distribution(sentiment_df):
    fig, ax = plt.subplots(figsize=(5, 3))  # smaller fixed size
    # Map colors based on sentiment
    color_map = {
        "Positive": "green",
        "Negative": "red",
        "Neutral": "#FF8C00"
    }
    colors = sentiment_df["Sentiment"].map(lambda x: color_map.get(x, "blue"))  # default blue if unknown
    ax.bar(sentiment_df["Sentiment"], sentiment_df["Count"], color=colors)
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)

########################################################################################

def show():
      # ---------------- Setup ----------------
    api_key = st.session_state.get("api_key", "")
    headers = {"x-api-key": api_key}
    # ---------------- Page: Home ----------------
    if st.session_state.page == "Home":
        st.title("📊 SentimentAI Dashboard")

        # ---------------- Text Prediction ----------------
        st.subheader("🔹 Try with Text")
        user_input = st.text_area("Type or paste text here...", height=120)

        if st.button("Predict"):
            if user_input.strip():
                try:
                    response = requests.post(
                        PREDICT_API,
                        json={"text": user_input},
                        headers=headers,
                        timeout=10
                    )
                    if response.status_code == 200:
                        st.session_state.prediction_result = response.json()
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"⚠️ API Error: {e}")
            else:
                st.warning("Please enter text.")

        # Show prediction result if available
        if st.session_state.prediction_result:
            data = st.session_state.prediction_result
            st.success("Prediction Complete ✅")

            sentiment = data.get("prediction", "N/A")
            confidence = round(data.get("Confidence", 0) * 100, 2)

            # Choose color based on sentiment
            if sentiment.lower() == "positive":
                color = "green"
            elif sentiment.lower() == "negative":
                color = "red"
            elif sentiment.lower() == "neutral":
                color = "#FF8C00"
            else:
                color = "black"

            col1, col2 = st.columns(2)
            col1.markdown(f"<h3 style='color:{color};'>{sentiment}</h3>", unsafe_allow_html=True)
            col2.metric("Confidence", f"{confidence}%")

        df = pd.read_json(DB_FILE)
        st.subheader("🔹 Evaluate the sentiment of comments in your database for informed decision-making.")
        # ---------------- Filters ----------------
        clauses = ["Overall"]
        if "Clause_id" in df.columns:
            clauses += sorted(df["Clause_id"].dropna().unique().tolist())
        selected_clause = st.selectbox("Choose Clause ID", clauses)

        stakeholders = ["Overall"]
        if "Stakeholders" in df.columns:
            stakeholders += sorted(df["Stakeholders"].dropna().unique().tolist())
        selected_stakeholder = st.selectbox("Choose Stakeholder", stakeholders)

        labels = ["Overall"]
        if "Label" in df.columns:
            labels += sorted(df["Label"].dropna().unique().tolist())
        selected_label = st.selectbox("Choose Label", labels)

        # ---------------- Run Analysis ----------------
        if st.button("Run Analysis"):
            with st.spinner("📊 Analyzing..."):
                df_filtered = df.copy()
                if selected_clause != "Overall":
                    df_filtered = df_filtered[df_filtered["Clause_id"] == selected_clause]
                if selected_stakeholder != "Overall":
                    df_filtered = df_filtered[df_filtered["Stakeholders"] == selected_stakeholder]
                if selected_label != "Overall":
                    df_filtered = df_filtered[df_filtered["Label"] == selected_label]

                if df_filtered.empty:
                    st.warning("⚠️ No data found in the Database.")
                    st.stop()

                # Send to backend
                csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
                files = {"file": ("filtered.csv", io.BytesIO(csv_bytes), "text/csv")}

                try:
                    resp = requests.post(ANALYZE_API, files=files, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    st.error(f"API Error: {e}")
                    st.stop()

            st.success("✅ Analysis Complete!")

            # ---------------- Top Statistics ----------------
            st.title("📊 Top Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Comments", data.get("total_comments", 0))
            col2.metric("Unique Clause", data.get("unique_clause", 0))
            col3.metric("Avg. Word Count", round(data.get("avg_word_count", 0), 2))

            # ---------------- Sentiment Distribution ----------------
            if "sentiment_counts" in data and data["sentiment_counts"]:
                st.title("📈 Sentiment Distribution")
                sentiment_df = pd.DataFrame(list(data["sentiment_counts"].items()), columns=["Sentiment", "Count"])
                
                # Define colors for sentiments
                sentiment_colors = {
                    "positive": "green",
                    "negative": "red",
                    "neutral": "#FF8C00"
                }
                colors = [sentiment_colors.get(s.lower(), "blue") for s in sentiment_df["Sentiment"]]

                col1, col2 = st.columns(2)
                
                # Bar chart
                with col1:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.bar(sentiment_df["Sentiment"], sentiment_df["Count"], color=colors)
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Count")
                    ax.set_title("Sentiment Counts")
                    plt.xticks(rotation=30)
                    st.pyplot(fig)
                
                # Pie chart
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.pie(sentiment_df["Count"], labels=sentiment_df["Sentiment"], autopct="%0.1f%%", colors=colors)
                    ax.axis("equal")
                    ax.set_title("Sentiment Proportions")
                    st.pyplot(fig)
            else:
                st.warning("⚠️ No sentiment distribution data available.")

            # ---------------- Word Cloud ----------------
            if data.get("wordcloud_base64"):
                st.title("☁️ Word Cloud")
                img_b64 = data["wordcloud_base64"]
                img = base64.b64decode(img_b64)
                st.image(img, use_container_width=True)  # Keeps it responsive

            # ---------------- Most Common Words ----------------
            if "top_words" in data:
                st.title("🔠 Most Common Words")
                top_words = pd.DataFrame(data["top_words"], columns=["Word", "Frequency"])
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.barh(top_words["Word"].head(15), top_words["Frequency"].head(15), color="orange")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)

            # ---------------- N-Grams ----------------
            if "top_bigrams" in data or "top_trigrams" in data:
                st.title("📌 N-Gram Analysis")
                col1, col2 = st.columns(2)

                if "top_bigrams" in data:
                    with col1:
                        st.subheader("Top Bigrams")
                        bigram_df = pd.DataFrame(data["top_bigrams"], columns=["Bigram", "Frequency"])
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.barh(bigram_df["Bigram"].head(10), bigram_df["Frequency"].head(10), color="blue")
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)

                if "top_trigrams" in data:
                    with col2:
                        st.subheader("Top Trigrams")
                        trigram_df = pd.DataFrame(data["top_trigrams"], columns=["Trigram", "Frequency"])
                        fig, ax = plt.subplots(figsize=(4, 3))
                        ax.barh(trigram_df["Trigram"].head(10), trigram_df["Frequency"].head(10), color="purple")
                        ax.invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)

            # ---------------- Clause Sentiment Bar Chart ----------------
            if "clause_sentiment" in data and data["clause_sentiment"]:
                clause_data = data["clause_sentiment"]

                if isinstance(clause_data, dict) and len(clause_data) > 0:
                    st.title("📊 Sentiment % by Clause ID")

                    clause_df = pd.DataFrame(clause_data).T
                    clause_df = clause_df[["Positive", "Neutral", "Negative"]]

                    clause_mapping = dict(zip(df["Clause"], df["Clause_id"]))
                    clause_df.index = clause_df.index.map(lambda x: clause_mapping.get(x, x))

                    clause_df = clause_df.sort_index(key=lambda x: x.astype(str).astype(float))

                    n_clauses = len(clause_df)
                    index = np.arange(n_clauses)
                    bar_width = 0.25

                    fig, ax = plt.subplots(figsize=(6, max(3, n_clauses * 0.3)))
                    ax.barh(index - bar_width, clause_df["Positive"], height=bar_width, label="Positive", color="green")
                    ax.barh(index, clause_df["Neutral"], height=bar_width, label="Neutral", color="#FF8C00")
                    ax.barh(index + bar_width, clause_df["Negative"], height=bar_width, label="Negative", color="red")

                    ax.set_yticks(index)
                    ax.set_yticklabels(clause_df.index)
                    ax.set_xlabel("Percentage (%)")
                    ax.set_title("Sentiment Distribution per Clause ID")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)

            # ---------------- Word Count Distribution ----------------
            if "word_count_data" in data:
                st.title("📊 Word Count Distribution by Sentiment")
                wc_df = pd.DataFrame(data["word_count_data"])
                if not wc_df.empty:
                    create_kde_by_category(wc_df, 'word_count', 'category', 'Word Count Distribution by Sentiment Category', figsize=(5,3))

            # ---------------- Top Words by Sentiment ----------------
            if "clean_comments" in data:
                comments_df = pd.DataFrame(data["clean_comments"])
                if not comments_df.empty:
                    st.title("🔠 Top Words by Sentiment Category")
                    plot_top_words_by_category(comments_df, 'clean_comment', 'category', n=20, figsize=(6,4))
