import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils.config import DB_PATH

def create_collection(collection_name: str, 
                     documents: List[Document], 
                     embedding_model: Embeddings) -> Chroma:
    """
    Yeni bir Chroma vektör koleksiyonu oluşturur ve belgeleri ekler.
    """
    # Koleksiyon için tam dizin yolu
    collection_path = os.path.join(DB_PATH, collection_name)
    os.makedirs(collection_path, exist_ok=True)
    
    # Chroma veritabanını oluştur ve belgeleri ekle
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=collection_path,
        collection_name=collection_name
    )
    
    # Değişiklikleri diske kaydet
    vectordb.persist()
    
    return vectordb

def load_collection(collection_name: str, 
                   embedding_model: Embeddings) -> Chroma:
    """
    Varolan bir Chroma koleksiyonunu yükler.
    """
    collection_path = os.path.join(DB_PATH, collection_name)
    
    # Koleksiyon yoksa hata döndür
    if not os.path.exists(collection_path):
        raise FileNotFoundError(f"Collection not found: {collection_path}")
    
    # Chroma veritabanını yükle
    vectordb = Chroma(
        persist_directory=collection_path,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    
    return vectordb

def add_documents_to_collection(vectordb: Chroma, 
                               documents: List[Document]) -> Chroma:
    """
    Varolan bir koleksiyona yeni belgeler ekler.
    """
    # Belgeleri vektör veritabanına ekle
    vectordb.add_documents(documents)
    
    # Değişiklikleri diske kaydet
    vectordb.persist()
    
    return vectordb

def delete_documents_from_collection(vectordb: Chroma, 
                                    filter_dict: Dict[str, Any]) -> None:
    """
    Belirli bir filtreye göre belgeleri koleksiyondan siler.
    """
    # Filtreye göre belgeleri sil
    vectordb.delete(where=filter_dict)
    
    # Değişiklikleri diske kaydet
    vectordb.persist()

def get_qa_chain(llm: BaseLanguageModel,
                retriever,
                system_prompt: str,
                chain_type: str = "stuff") -> RetrievalQA:
    """
    Belirli bir LLM ve retriever ile soru-cevap zinciri oluşturur.
    """
    # Geriye dönük uyumluluk için eski RetrievalQA yöntemini kullanıyoruz
    # Prompt şablonu oluştur
    prompt_template = f"""
    {system_prompt}
    
    Use the following text to answer the question:
    {{context}}
    
    Question: {{question}}
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # RetrievalQA zincirini oluştur
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

def get_relevant_docs(retriever, query: str, k: int = 4) -> List[Document]:
    """
    Verilen sorgu için en alakalı belge parçalarını döndürür.
    """
    return retriever.get_relevant_documents(query, k=k) 