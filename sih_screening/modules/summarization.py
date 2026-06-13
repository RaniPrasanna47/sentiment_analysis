import streamlit as st
import pandas as pd
import requests
import  os

DB_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__),"..", "..", "database", "virtual_db.json")
)

SUMMARIZE_API = "http://127.0.0.1:8500/summarize"

def show():
    st.title("📝 Summarization")
    # long_text = st.text_area("Enter text to summarize:", height=300)
    # if st.button("Summarize"):
    #     if long_text.strip() == "":
    #         st.warning("Please enter text!")
    #     else:
    #         try:
    #             response = requests.post("http://127.0.0.1:8500/summarize", json={"text": long_text})
    #             if response.status_code==200:
    #                 st.subheader("Summary:")
    #                 st.write(response.json().get("summary",""))
    #             else:
    #                 st.error("Summarization API error")
    #         except Exception as e:
    #             st.error(f"Error: {e}")

        # ---------------- Filters ----------------

        # Read directly from JSON file
    df = pd.read_json(DB_FILE)
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

        # ---------------- sumarization Analysis ----------------
    if st.button("Summarize comment"):
        with st.spinner("📊 sumarization..."):
            df_filtered = df.copy()
            if selected_clause != "Overall":
                df_filtered = df_filtered[df_filtered["Clause_id"] == selected_clause]
            if selected_stakeholder != "Overall":
                df_filtered = df_filtered[df_filtered["Stakeholders"] == selected_stakeholder]
            if selected_label != "Overall":
                df_filtered = df_filtered[df_filtered["Label"] == selected_label]

            if df_filtered.empty:
                st.warning("⚠️ No data found for the selected Clause/Stakeholder/Label.")
                st.stop()

            # Merge all comments into one long string separated by "."
            long_text = ". ".join(df_filtered["Comment"].astype(str))
            if long_text.strip() == "":
                st.warning("Your Database have no commnet!")
            else:
                try:
                    response = requests.post(SUMMARIZE_API, json={"text": long_text})
                    if response.status_code==200:
                        # st.write(response.json().get("summary",""))
                        summary_text = response.json().get("summary", "")
                        # Display in a text area
                        st.subheader("📄 Combined Comments")
                        long_text_input = st.text_area(
                            label="All comments merged into one text",
                            value=long_text,
                            height=200
                            )
                        if long_text.strip() == "":
                            st.warning("Please enter text!")
                        st.subheader("Summary:")
                        summary_text_input = st.text_area(
                            label="Summary of the text",
                            value=summary_text,
                            height=70
                            )

                    else:
                        st.error("Summarization API error")
                except Exception as e:
                    st.error(f"Error: {e}")