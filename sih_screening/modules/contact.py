import streamlit as st

def show():
    st.title("📞 Contact")
    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("Thank you! Message submitted.")
    st.markdown("**Support Email:** youremail@example.com")
    st.markdown("Connect on [LinkedIn](https://linkedin.com) | [GitHub](https://github.com/your-repo)")
