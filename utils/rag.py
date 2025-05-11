import os
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import numpy as np
from utils.config import DB_PATH

# Kelime ağırlıklandırma için stop words listesi
STOP_WORDS = set([
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "with", "by", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "from", "up", "down", "of", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "have", "has", "had", "having", "do", "does", "did", "doing", "would",
    "could", "should", "ought", "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've",
    "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd", "i'll", "you'll",
    "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
    "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot",
    "couldn't", "mustn't", "let's", "that's", "who's", "what's", "here's", "there's", "when's", "where's",
    "why's", "how's"
])

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
    
    # Değişiklikleri diske kaydet - artık gerekli değil
    # vectordb.persist()
    
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
    
    # Değişiklikleri diske kaydet - artık gerekli değil
    # vectordb.persist()
    
    return vectordb

def delete_documents_from_collection(vectordb: Chroma, 
                                    filter_dict: Dict[str, Any]) -> None:
    """
    Belirli bir filtreye göre belgeleri koleksiyondan siler.
    """
    # Filtreye göre belgeleri sil
    vectordb.delete(where=filter_dict)
    
    # Değişiklikleri diske kaydet - artık gerekli değil
    # vectordb.persist()

def extract_keywords(query: str) -> List[Tuple[str, float]]:
    """
    Sorgudan anahtar kelimeleri çıkarır ve ağırlıklandırır.
    
    Args:
        query: Kullanıcı sorgusu
        
    Returns:
        Anahtar kelime ve ağırlık çiftlerinin listesi
    """
    # Metni temizle ve kelimelere ayır
    query = query.lower()
    words = re.findall(r'\b\w+\b', query)
    
    # Stop words olmayan kelimeleri filtrele
    keywords = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    
    # Her kelime için tekrar sayısı hesapla
    word_counts = {}
    for word in keywords:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Sorguda geçen kelimeleri ağırlıklandır
    weighted_keywords = []
    for word, count in word_counts.items():
        # Tekrar eden kelimeler daha yüksek ağırlık alır
        # Daha uzun kelimeler genellikle daha önemlidir
        weight = count * (1 + 0.1 * len(word))
        weighted_keywords.append((word, weight))
    
    # Ağırlığa göre sırala
    weighted_keywords.sort(key=lambda x: x[1], reverse=True)
    
    return weighted_keywords

def create_weighted_retriever(vectordb: Chroma, search_kwargs: Optional[Dict[str, Any]] = None):
    """
    Ağırlıklandırılmış sorgu yapabilen özel bir retriever oluşturur.
    
    Args:
        vectordb: Chroma vektör veritabanı
        search_kwargs: Arama parametreleri
        
    Returns:
        Ağırlıklandırılmış retriever
    """
    # Varsayılan arama parametreleri
    if search_kwargs is None:
        search_kwargs = {"k": 4}
    
    # Standart retriever'ı oluştur
    standard_retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    
    # Ağırlıklandırılmış retriever
    class WeightedRetriever:
        def __init__(self, retriever, vectorstore):
            self.retriever = retriever
            self.vectorstore = vectorstore
            self._kwargs = {"k": 4}
            self.search_type = "weighted"  # retriever özelliği için
            
        def get_relevant_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
            # Anahtar kelimeleri çıkar
            keywords = extract_keywords(query)
            
            # Eğer anahtar kelime bulunamazsa standart retriever'ı kullan
            if not keywords:
                return self.retriever.get_relevant_documents(query, k=k)
            
            # Ağırlıklı sorgu için ayarlar
            k_val = k if k is not None else self._kwargs.get("k", 4)
            
            # Anahtar kelimeler için de sonuçlar getir (toplam k'dan fazla)
            expanded_k = min(k_val * 2, 8)  # Maksimum 8 sonuç
            
            # Ana sorgudan sonuçları al
            main_results = self.retriever.get_relevant_documents(query, k=expanded_k)
            
            # Sonuçları değerlendirmek için puan matrisi oluştur
            doc_scores = {doc.page_content: 1.0 for doc in main_results}
            
            # En önemli 3 anahtar kelime için ek sorgular yap
            for keyword, weight in keywords[:3]:
                # Her anahtar kelime için ağırlığına göre sonuç getir
                keyword_results = self.retriever.get_relevant_documents(keyword, k=2)
                
                # Bu sonuçların puanlarını anahtar kelime ağırlığına göre artır
                for doc in keyword_results:
                    if doc.page_content in doc_scores:
                        doc_scores[doc.page_content] += 0.5 * weight
                    else:
                        doc_scores[doc.page_content] = 0.5 * weight
            
            # Tüm belgeleri puanlarına göre sırala
            scored_docs = []
            for doc in main_results:
                if doc.page_content in doc_scores:
                    scored_docs.append((doc, doc_scores[doc.page_content]))
            
            # Puanlarına göre sırala
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # En yüksek puanlı k belgeyi döndür
            return [doc for doc, _ in scored_docs[:k_val]]
        
        # LangChain tarafından beklenen metotlar
        def with_config(self, **kwargs):
            # LLM ya da benzeri yapıların arama yapılandırmasını değiştirmesi için
            self._kwargs.update(kwargs)
            return self
            
        def invoke(self, input):
            # Yeni LangChain API'si için invoke metodu
            if isinstance(input, str):
                return self.get_relevant_documents(input)
            elif isinstance(input, dict) and "query" in input:
                return self.get_relevant_documents(input["query"])
            else:
                raise ValueError(f"Invalid input: {input}")
        
        # Ek LangChain uyumluluk metotları
        def get_scores(self, docs, query):
            # Belgeler için skor sağlayan bir metot
            # Basit bir örnek, gerçek skorlama daha karmaşık olabilir
            return [1.0] * len(docs)  # Tüm belgelere 1.0 skor ver
            
        @property
        def metadata_keys(self):
            # Metadata bilgisi
            try:
                return self.vectorstore.metadata_keys
            except:
                return set()
            
        def as_retriever(self, **kwargs):
            # Kendi türünden bir retriever döndür
            self._kwargs.update(kwargs)
            return self
    
    # Weighted Retriever örneği oluştur
    return WeightedRetriever(standard_retriever, vectordb)

def get_qa_chain(llm: BaseLanguageModel,
                retriever,
                system_prompt: str,
                chain_type: str = "stuff"):
    """
    Belirli bir LLM ve retriever ile soru-cevap zinciri oluşturur.
    Modern LangChain 0.1+ API'sini kullanır.
    """
    # Prompt şablonu oluştur
    prompt_template = f"""
    {system_prompt}
    
    Use the following text to answer the question:
    {{context}}
    
    Question: {{input}}
    Answer:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )
    
    # Doküman zinciri oluştur
    qa_documents_chain = create_stuff_documents_chain(llm, PROMPT)
    
    # Retrieval zincirini oluştur - daha verimli yapı
    qa_chain = create_retrieval_chain(retriever, qa_documents_chain)
    
    # Optimize edilmiş sarmalayıcı fonksiyon
    def qa_chain_wrapper(inputs):
        # Doğrudan sorguyu al ve zincire gönder
        result = qa_chain.invoke({"input": inputs["query"]})
        # Sonucu dönüştür
        return {
            "result": result["answer"],
            "source_documents": result["context"]
        }
    
    return qa_chain_wrapper

def get_relevant_docs(retriever, query: str, k: int = 4) -> List[Document]:
    """
    Verilen sorgu için en alakalı belge parçalarını döndürür.
    """
    return retriever.get_relevant_documents(query, k=k) 