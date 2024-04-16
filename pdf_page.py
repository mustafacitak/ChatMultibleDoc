import os
import streamlit as st
from PyPDF2 import PdfReader
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from streamlit_pdf_viewer import pdf_viewer
from streamlit import session_state as ss

# OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"

def create_or_load_vector_store(text, vector_base_name):
    """
    Create or load vector store from given text.
    """
    if os.path.exists(vector_base_name):
        with open(vector_base_name, "rb") as f:
            VectorStore = pickle.load(f)
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Save the vector store
        with open(vector_base_name, "wb") as f:
            pickle.dump(VectorStore, f)

    return VectorStore

def pdf_page():
    st.title("İstediğin Doküman Formatı İle Konuş")

    # Yuklenen dosyaların listesi
    uploadedFiles = st.sidebar.file_uploader("Upload your files.", type=['pdf'], accept_multiple_files=True)

    # Vektör tabanı dosya adı
    vector_base_name = "vector_base.pkl"

    # Kullanıcı sorusu ve cevap bölümü
    if uploadedFiles:
        st.subheader("Chat:")
        query = st.text_input("Ask questions about your file:")
        if st.button("Send"):
            if query:
                for file in uploadedFiles:
                    pdf_reader = PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                    VectorStore = create_or_load_vector_store(text, vector_base_name)

                    # Benzer dokümanların getirilmesi
                    k = 10  # Number of nearest neighbors to retrieve
                    distances = []  # List to store the distances
                    labels = []
                    docs = VectorStore.similarity_search(
                        query=query, k=k, distances=distances, labels=labels)

                    # Chat modelinin yüklenmesi ve dokümanla konuşulması
                    llm = ChatOpenAI(temperature=0.07, model_name="gpt-3.5-turbo")
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=query)

                    # Kullanıcı girdisi ve cevabı görüntüleme
                    st.subheader("Answer:")
                    st.write(response)

    # Yuklenen belgelerin içeriği ve işlenmesi
    if uploadedFiles:
        st.subheader("File Preview:")
        for file in uploadedFiles:
            st.write(file.name)
            pdf_reader = PdfReader(file)
            binary_data = file.getvalue()
            pdf_viewer(input=binary_data, width=700)

