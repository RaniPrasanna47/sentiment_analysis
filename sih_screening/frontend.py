import streamlit as st
import requests
import pandas as pd
import numpy as np
import io
from st_aggrid import AgGrid
import matplotlib.pyplot as plt
import base64
from wordcloud import WordCloud
import seaborn as sns
from collections import Counter
from modules import about, history, summarization, contact,Upload, home, Live_Graph


# ---------------- Page Config ----------------
st.set_page_config(
    layout="wide",
    page_title="SentimentAI",
    initial_sidebar_state="collapsed"
)

# Hide default Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---------------- API URLs ----------------
CLASSIFY_API = "http://127.0.0.1:8500/classify_file"

# ---------------- Session State ----------------
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "key_verified" not in st.session_state:
    st.session_state.key_verified = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "classified_df" not in st.session_state:
    st.session_state.classified_df = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# ---------------- Session State ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

def set_page(page_name):
    st.session_state.page = page_name

# ---------------- Custom CSS for Navbar Buttons ----------------
st.markdown("""
<style>
.navbar-button {
    width: 120px !important;  /* Fixed width for all buttons */
    height: 45px;
    font-size: 16px;
    font-weight: 500;
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    border: none;
    transition: background-color 0.3s ease;
    margin-right: 5px;  /* small spacing between buttons */
}
.navbar-button:hover {
    background-color: #0d5a8c;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Top Navbar (Left-Aligned) ----------------
cols = st.columns([0.01, 0.01, 0.015, 0.01, 0.01, 0.01,0.01])  # small fractions to keep them left aligned

with cols[0]:
    if st.button("🏠 Home", key="home", on_click=lambda: set_page("Home")):
        pass
with cols[1]:
    if st.button("ℹ️ About", key="about", on_click=lambda: set_page("About")):
        pass
with cols[2]:
    if st.button("Summarization", key="summarization", on_click=lambda: set_page("Summarization")):
        pass
with cols[3]:
    if st.button("Upload", key="upload", on_click=lambda: set_page("Upload")):
        pass
with cols[4]:
    if st.button("Live Graph", key="Live_Graph", on_click=lambda: set_page("Live_Graph")):
        pass
with cols[5]:
    if st.button("History", key="history", on_click=lambda: set_page("History")):
        pass
with cols[6]:
    if st.button("📞 Contact", key="contact", on_click=lambda: set_page("Contact")):
        pass


# ---------------- Sidebar ----------------
st.sidebar.title("🔑 SentimentAI – Secure Access")

# Always show API key input
api_key_input = st.sidebar.text_input(
    "Enter your API Key",
    type="password",
    placeholder="Enter your API key here..."
)

if st.sidebar.button("Submit API Key"):
    try:
        response = requests.post(
            CLASSIFY_API,
            files={},
            headers={"x-api-key": api_key_input},
            timeout=5
        )
        if response.status_code == 401:
            st.sidebar.error("❌ Invalid API key. Please try again.")
        else:
            st.session_state.api_key = api_key_input
            st.session_state.key_verified = True
            st.sidebar.success("✅ API key verified successfully!")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"⚠️ Unable to verify API key: {e}")

if not st.session_state.key_verified:
    st.warning("🔒 Please enter and submit your API key to unlock the app.")
    st.stop()

api_key = st.session_state.api_key
headers = {"x-api-key": api_key}


# ---------------- Page Rendering ----------------
if st.session_state.page == "Home":
    home.show()
elif st.session_state.page == "About":
    about.show()
elif st.session_state.page == "History":
    history.show()
elif st.session_state.page == "Upload":
    Upload.show()
elif st.session_state.page == "Live_Graph":
    Live_Graph.show()
elif st.session_state.page == "Summarization":
    summarization.show()
elif st.session_state.page == "Contact":
    contact.show()
else:
    st.write("Unknown page selected.")
