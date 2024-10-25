import streamlit as st

# Uygulama başlarken sohbet geçmişi yoksa bir boş liste olarak tanımla
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Kullanıcıdan mesaj alacak bir giriş alanı
user_message = st.text_input("Mesajınızı girin:")

# Kullanıcı 'Gönder' butonuna bastığında mesajı sohbet geçmişine ekle
if st.button("Gönder"):
    if user_message:
        # Mesajı sohbet geçmişine ekle
        st.session_state.messages.append({"role": "user", "content": user_message})
        # Yapay zekadan gelen yanıtı ekle (örnek olarak basit bir yanıt kullanıyoruz)
        st.session_state.messages.append({"role": "assistant", "content": f"Cevabınız: {user_message}"})

# Sohbet geçmişini ekranda göster
st.write("### Sohbet Geçmişi:")
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"👤 **Siz**: {message['content']}")
    else:
        st.write(f"🤖 **Asistan**: {message['content']}")

