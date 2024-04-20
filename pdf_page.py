import os
import streamlit as st
import pandas as pd
import docx
from PyPDF2 import PdfReader
import pickle
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from streamlit_pdf_viewer import pdf_viewer

os.environ["OPENAI_API_KEY"] = "sk-proj-ugbMH2Kt5r4gSJCGsHFQT3BlbkFJhIaQPQ7uIVBmx7fdR96H"
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.error("OpenAI API key is missing. Please make sure to set it in your .env file.")

# Vektör tabanı ve chat modelini bir kez yükle
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=len
)

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0.07, model_name="gpt-3.5-turbo")
chain = load_qa_chain(llm=llm, chain_type="stuff")


def pdf_page():
    column1, column2 = st.columns(2)

    st.markdown("""
        <style>
            .st-emotion-cache-gh2jqd {
                max-width: none !important;
                padding: 2rem !important;
            }
        </style>
        """, unsafe_allow_html=True)
    vanelli_logo = "./images/vanelli.svg"
    st.sidebar.image(vanelli_logo, width=200)
    uploadedFiles = st.sidebar.file_uploader("", type=['pdf', '.csv', '.xlsx', '.xls', '.docx'],
                                             accept_multiple_files=True)

    with column1:
        st.header("Dokümanla Chat")
    # Yüklenen dosyaların işlenmesi ve vektör tabanının oluşturulması
        if uploadedFiles:
            text = ""
            for file in uploadedFiles:
                extension = file.name[len(file.name) - 3:]
                if (extension == "pdf"):
                    file_reader = PdfReader(file)
                    for page in file_reader.pages:
                        text += page.extract_text()
                elif (extension == "csv"):
                    file_reader = pd.read_csv(file)
                    text += "\n".join(
                        file_reader.apply(lambda row: ', '.join(row.values.astype(str)), axis=1))
                elif (extension == "lsx" or extension == "xls"):
                    file_reader = pd.read_excel(file)
                    text += "\n".join(
                        file_reader.apply(lambda row: ', '.join(row.values.astype(str)), axis=1))
                elif (extension == "ocx"):
                    file_reader = docx.Document(file)
                    list = [paragraph.text for paragraph in file_reader.paragraphs]
                    text += ' '.join(list)

            if (text):
                chunks = text_splitter.split_text(text=text)
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

                # Vektör tabanının kaydedilmesi
                vector_base_name = "vector_base.pkl"
                with open(vector_base_name, "wb") as f:
                    pickle.dump(VectorStore, f)

            else:
                st.error("Bir şeyler yanlış gitti")

        # Chat modelini yükle ve kullanıcı ile etkileşime geç
        if uploadedFiles:
            st.write("Dokümanlara Ne Sormak İstiyorsunuz?")
            query = st.text_input( label= "Soruyu aşağıya yazınız ve enter tuşuna basımız.")
            st.button("Gönder")

            # Enter tuşuna basıldığında
            if query and os.path.exists(vector_base_name):
                with open(vector_base_name, "rb") as f:
                    VectorStore = pickle.load(f)

                    # Benzer dokümanların getirilmesi
                    k = 10  # Number of nearest neighbors to retrieve
                    distances = []  # List to store the distances
                    labels = []
                    docs = VectorStore.similarity_search(
                        query=query, k=k, distances=distances, labels=labels)

                    # Chat modelinin yüklenmesi ve dokümanla konuşulması
                    response = chain.run(input_documents=docs, question=query)
                    st.divider()
                    st.subheader("Cevap: ")
                    st.write(response)
                    st.divider()

        else:
            st.write("Merhaba! Doküman yüklemediniz. Lütfen doküman yükleyin.")

    # Sağ sütun için ayrı işlemler
    with column2:
    # Yuklenen belgelerin içeriği
        if uploadedFiles:
            st.subheader("Yüklenen Dokümanlar:")
            for file in uploadedFiles:
                st.write(file.name)
                file_extension = file.name.split(".")[-1]  # Dosya uzantısını al
                if file_extension == "pdf":
                    pdf_reader = PdfReader(file)
                    binary_data = file.getvalue()
                    pdf_viewer(input=binary_data, width=500, height=550)
                elif file_extension == "xlsx":
                    df = pd.read_excel(file)
                    st.write(df)
                elif file_extension == "csv":
                    try:
                        df = pd.read_csv(file)
                        st.write(df)
                    except pd.errors.EmptyDataError:
                        st.write("Dosya Yükleme Başarılı.")
                elif file_extension == "docx":
                    docx_reader = docx.Document(file)
                    for paragraph in docx_reader.paragraphs:
                        st.write(paragraph.text)
                else:
                    st.write("Dosya Türü Desteklenmiyor!")