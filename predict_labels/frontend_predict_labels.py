import streamlit as st
import requests
import pandas as pd
import io

# API endpoints
PREDICT_URL = "http://127.0.0.1:8000/predict"
UPDATE_URL = "http://127.0.0.1:8000/update_model"

st.title("DistilBERT Sentiment Analysis & Live Update")

# ---------------- Prediction Section ----------------
st.header("Real-time Prediction")
user_input = st.text_area("Enter text for prediction:")

if st.button("Predict"):
    if user_input.strip() != "":
        response = requests.post(PREDICT_URL, json={"text": user_input})
        if response.status_code == 200:
            resp_json = response.json()
            prediction = resp_json.get("prediction")
            Confidence = resp_json.get("Confidence")  # make sure your API actually returns "Score"
            st.success(f"Prediction: {prediction}, Score: {Confidence}")
        else:
            st.error("Error in API request")
    else:
        st.warning("Please enter some text.")

# # ---------------- Model Update Section ----------------
# st.header("Update Model with New CSV")
# uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# if st.button("Update Model"):
#     if uploaded_file is not None:
#         files = {"file": uploaded_file.getvalue()}
#         response = requests.post(UPDATE_URL, files={"file": uploaded_file})
#         if response.status_code == 200:
#             st.success(response.json()["status"])
#         else:
#             st.error("Error updating model. Check CSV format.")
#     else:
#         st.warning("Please upload a CSV file with 'Comment' and 'Label' columns.")


st.title("CSV Sentiment Classifier")

uploaded_file = st.file_uploader("Upload your CSV/ Excel file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded CSV/Excel:")
    st.dataframe(df.head())

    if st.button("Generate Labeled CSV/Excel File"):
        # Reset file pointer
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
        
        try:
            response = requests.post("http://127.0.0.1:8000/classify_file", files={"file": uploaded_file})
            
            if response.status_code == 200:
                st.success("CSV classified successfully!")
                st.download_button(
                    label="Download Labeled CSV",
                    data=response.content,
                    file_name="labeled_data.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"Error classifying CSV: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
