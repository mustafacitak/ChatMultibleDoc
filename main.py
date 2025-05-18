import streamlit as st

st.set_page_config(
    page_title="DoChat v0.3",
    page_icon="💬",
)

# Çoklu sayfa navigation
pages = [
    st.Page("pages/1_upload_document.py", title="Document", icon="📎"),
    st.Page("pages/2_chat.py", title="Chat", icon="💬"),
]

pg = st.navigation(pages)

pg.run() 