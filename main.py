import streamlit as st
from document_page import document_page

# Örnek kullanıcı adları ve parolaları
user_credentials = {
    "1": "1",
    "mail_adress@mail.com": "mail123",
    "mail_adress@mail.com": "mail123",
}

def login():
    st.title("Giriş")
    username = st.text_input("Kullanıcı Adı")
    password = st.text_input("Parola", type="password")
    if st.button("Giriş"):
        if username in user_credentials and user_credentials[username] == password:
            st.session_state.is_logged_in = True
            st.success("Giriş Başarılı!")
            st.experimental_rerun()
        else:
            st.error("Yanlış Kullanıcı Adı veya Parola.")

def main():
    if "is_logged_in" not in st.session_state:
        st.session_state.is_logged_in = False

    if not st.session_state.is_logged_in:
        login()
    else:
        document_page()

if __name__ == "__main__":
    main()
