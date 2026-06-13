import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json, os

# ----------------- Database Path -----------------
DB_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "database", "virtual_db.json")
)

# ----------------- Load JSON Data -----------------
@st.cache_data
def get_data():
    with open(DB_FILE, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    return df

# ----------------- Sentiment mapping -----------------
sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
color_map = {"Positive": "green", "Neutral": "yellow", "Negative": "red"}

# ----------------- Main Streamlit App -----------------
def show():
    st.set_page_config(layout="wide", page_title="Sentiment Trend Dashboard")
    st.title("📊 Sentiment Trend Dashboard")

    df = get_data()
    df["Sentiment_Value"] = df["Label"].map(sentiment_map)

    # ---------------- Filters ----------------
    clauses = ["Overall"]
    if "Clause_id" in df.columns:
        clauses += sorted(df["Clause_id"].dropna().unique().tolist())
    selected_clause = st.selectbox("Choose Clause ID", clauses)

    stakeholders = ["Overall"]
    if "Stakeholders" in df.columns:
        stakeholders += sorted(df["Stakeholders"].dropna().unique().tolist())
    selected_stakeholder = st.selectbox("Choose Stakeholder", stakeholders)

    # ---------------- Button ----------------
    if st.button("📈 Plot Graphs"):
        # Filter Logic
        filtered_df = df.copy()
        if selected_clause != "Overall":
            filtered_df = filtered_df[filtered_df["Clause_id"] == selected_clause]
        if selected_stakeholder != "Overall":
            filtered_df = filtered_df[filtered_df["Stakeholders"] == selected_stakeholder]

        filtered_df = filtered_df.sort_values("Datetime")

        # ---------------- Plotting ----------------
        if selected_clause == "Overall":
            # Each stakeholder → separate graph
            for stakeholder in filtered_df["Stakeholders"].unique():
                stakeholder_df = filtered_df[filtered_df["Stakeholders"] == stakeholder].sort_values("Datetime")
                fig = go.Figure()
                for sentiment in ["Positive", "Neutral", "Negative"]:
                    sentiment_df = stakeholder_df[stakeholder_df["Label"] == sentiment]
                    if not sentiment_df.empty:
                        fig.add_trace(go.Scatter(
                            x=sentiment_df["Datetime"],
                            y=sentiment_df["Clause_id"],
                            mode="lines+markers",
                            name=sentiment,
                            line=dict(color=color_map[sentiment], shape="linear", smoothing=1.3),
                            marker=dict(size=8),
                        ))
                fig.update_layout(
                    title=f"Stakeholder: {stakeholder} (All Clauses)",
                    xaxis_title="Date & Time",
                    yaxis_title="Clause ID",
                    hovermode="x unified",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            # Specific Clause ID → still break by stakeholder
            for stakeholder in filtered_df["Stakeholders"].unique():
                stakeholder_df = filtered_df[filtered_df["Stakeholders"] == stakeholder].sort_values("Datetime")
                fig = go.Figure()
                for sentiment in ["Positive", "Neutral", "Negative"]:
                    sentiment_df = stakeholder_df[stakeholder_df["Label"] == sentiment]
                    if not sentiment_df.empty:
                        fig.add_trace(go.Scatter(
                            x=sentiment_df["Datetime"],
                            y=sentiment_df["Sentiment_Value"],
                            mode="lines+markers",
                            name=sentiment,
                            line=dict(color=color_map[sentiment], shape="spline", smoothing=1.3),
                            marker=dict(size=8),
                        ))
                fig.update_layout(
                    title=f"Stakeholder: {stakeholder} (Clause {selected_clause})",
                    xaxis_title="Date & Time",
                    yaxis=dict(
                        tickvals=[-1, 0, 1],
                        ticktext=["Negative", "Neutral", "Positive"]
                    ),
                    yaxis_title="Sentiment Trend",
                    hovermode="x unified",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
