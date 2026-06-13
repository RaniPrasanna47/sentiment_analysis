import streamlit as st
import pandas as pd
import requests
import json, os
from st_aggrid import AgGrid, GridOptionsBuilder

def show():
    # ---------------- Setup ----------------
    api_key = st.session_state.get("api_key", "")
    headers = {"x-api-key": api_key}
    CLASSIFY_API = "http://127.0.0.1:8500/classify_file"
    DB_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "database/virtual_db.json"))

    st.subheader("📂 Upload CSV for Classification")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}

        if st.button("Upload CSV to DB"):
            try:
                with st.spinner("Processing file..."):
                    response = requests.post(CLASSIFY_API, files=files, headers=headers)
                    response.raise_for_status()
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                st.stop()

            # ---------------- Parse JSON ----------------
            try:
                resp_json = response.json()
            except json.JSONDecodeError as e:
                st.error(f"Failed to decode JSON: {e}")
                st.stop()

            if "data" not in resp_json or not isinstance(resp_json["data"], list):
                st.error("Backend response missing 'data'")
                st.stop()

            # Convert to DataFrame
            df = pd.DataFrame(resp_json["data"])

            # ---------------- Save to JSON DB without duplicates ----------------
            try:
                os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
                if not os.path.exists(DB_FILE) or os.stat(DB_FILE).st_size == 0:
                    with open(DB_FILE, "w") as f:
                        json.dump([], f)

                with open(DB_FILE, "r") as f:
                    try:
                        db_data = json.load(f)
                    except:
                        st.write("Try one time Again..")
                        db_data = []

                # Track existing user_names
                existing_usernames = {row.get("User_name") for row in db_data if "User_name" in row}
                new_entries = []

                # Add only new entries
                for _, row in df.iterrows():
                    if row.get("User_name") in existing_usernames:
                        continue  # Skip duplicates

                    entry = row.to_dict()
                    # Convert to native Python types for JSON
                    for k, v in entry.items():
                        if pd.isna(v):
                            entry[k] = None
                        elif hasattr(v, "item"):
                            entry[k] = v.item()
                        elif isinstance(v, (pd.Timestamp, pd.Timedelta)):
                            entry[k] = str(v)

                    db_data.append(entry)
                    new_entries.append(entry)
                    existing_usernames.add(row.get("user_name"))

                # Save updated DB
                with open(DB_FILE, "w") as f:
                    json.dump(db_data, f, indent=4)

                st.success(f"✅ {len(new_entries)} new entries added to database")
                st.info(f"📊 Total entries in database: {len(db_data)}")

            except Exception as e:
                st.error(f"Failed to save to DB: {e}")


            # ---------------- Show Database Preview ----------------
            if os.path.exists(DB_FILE):
                try:
                    with open(DB_FILE, "r") as f:
                        db_data = json.load(f)
                    
                    if db_data:
                        df_db = pd.DataFrame(db_data)
                        st.write("### 📊 Database Preview")
                        
                        # AgGrid display
                        AgGrid(
                            df_db,
                            fit_columns_on_grid_load=True,
                            allow_unsafe_jscode=True,
                            theme='streamlit'
                        )

                        # Download button
                        st.download_button(
                            label="📥 Download Database CSV",
                            data=df_db.to_csv(index=False).encode("utf-8"),
                            file_name="database.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("📁 Database is empty.")
                except Exception as e:
                    st.error(f"Failed to load database: {e}")
            else:
                st.info("📁 Database not created yet.")
