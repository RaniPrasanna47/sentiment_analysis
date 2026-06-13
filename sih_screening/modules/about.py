import streamlit as st

def show():
    st.title("ℹ️ About SentimentAI")
    st.markdown("""
    SentimentAI is a lightweight front-end to inspect text sentiment, classify CSVs of comments,
    and run aggregated analysis.
    """)
    st.markdown("### Project Goal")
    st.markdown("- Insights into text sentiment for single comments or batch CSV uploads.")
    st.markdown("### Technology Stack")
    st.markdown("- Python, Streamlit, Plotly, Pandas\n- VADER, Hugging Face Transformers")
