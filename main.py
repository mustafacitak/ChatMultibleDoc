import streamlit as st
from pdf_page import pdf_page

# Örnek kullanıcı adları ve parolaları
user_credentials = {
    "1": "1",
    "ecemturan@vaneli.com": "ecem123",
    "mike@example.com": "mike789"
}

def login():
    st.title("Giriş")
    username = st.text_input("Kullanıcı Adı")
    password = st.text_input("Parola", type="password")
    if st.button("Giriş"):
        if username in user_credentials and user_credentials[username] == password:
            st.session_state.is_logged_in = True
            st.success("Giriş Başarılı!")
        else:
            st.error("Yanlış Kullanıcı Adı veya Parola.")

def main():
    if "is_logged_in" not in st.session_state:
        st.session_state.is_logged_in = False

    if not st.session_state.is_logged_in:
        login()
    else:
        pdf_page()

if __name__ == "__main__":
    main()
