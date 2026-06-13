import streamlit as st
import pandas as pd

def show():
    st.title("📜 History & Analysis Trends")
    # data = {
    #     "Date": pd.date_range(start="2025-01-01", periods=5, freq="D"),
    #     "Sentiment": [0.2,0.5,-0.1,0.7,0.3],
    #     "Source": ["Twitter","Reddit","Twitter","Reddit","Twitter"]
    # }
    # df = pd.DataFrame(data)
    # with st.expander("Filters"):
    #     start_date = st.date_input("Start Date", df["Date"].min())
    #     end_date = st.date_input("End Date", df["Date"].max())
    #     source_filter = st.selectbox("Source", ["All"] + df["Source"].unique().tolist())
    # filtered_df = df[(df["Date"]>=start_date) & (df["Date"]<=end_date)]
    # if source_filter != "All":
    #     filtered_df = filtered_df[filtered_df["Source"]==source_filter]
    # st.subheader("History Table")
    # st.dataframe(filtered_df)
