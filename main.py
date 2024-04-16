import streamlit as st
from pdf_page import pdf_page

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in user_credentials and user_credentials[username] == password:
            st.success("Login successful!")
            st.session_state.is_logged_in = True  # Giriş başarılı olduğunda is_logged_in'i True yap
            return True
        else:
            st.error("Invalid username or password.")
            return False

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Login", "PDF Page"])

    if page == "Login":
        login()
    elif page == "PDF Page":
        if not st.session_state.is_logged_in:
            st.warning("You need to login to access this page.")
        else:
            pdf_page()

# Örnek kullanıcı adları ve parolaları
user_credentials = {
    "candanfatih.com": "john123",
    "emma@example.com": "emma456",
    "mike@example.com": "mike789"
}

if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

if __name__ == "__main__":
    main()
