import os
import re
import unicodedata
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from utils.config import DB_PATH

def normalize_collection_name(name: str) -> str:
    """
    Koleksiyon adını ChromaDB'nin kabul edeceği bir formata dönüştürür.
    
    1. ASCII olmayan karakterleri benzer ASCII karakterlere dönüştürür (ör: ı -> i)
    2. Alfanumerik olmayan karakterleri altkareye dönüştürür (-)
    3. Başındaki ve sonundaki alfanumerik olmayan karakterleri kaldırır
    4. İzin verilen karakterleri kontrol eder
    
    Args:
        name: Normalize edilecek koleksiyon adı
        
    Returns:
        ChromaDB tarafından kabul edilebilir koleksiyon adı
    """
    # Önce unicodedata ile ASCII olmayan karakterleri benzer ASCII karakterlere dönüştür
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    
    # Sadece alfanumerik, alt çizgi ve tire karakterlerini tut
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    
    # Başlangıç ve sondaki alfanumerik olmayan karakterleri kaldır
    name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', name)
    
    # Eğer adın uzunluğu 3 karakterden az ise, "col_" öneki ekle
    if len(name) < 3:
        name = "col_" + name
    
    # Eğer adın uzunluğu 63 karakterden fazla ise, kes
    if len(name) > 63:
        name = name[:63]
    
    # Eğer son karakter alfanumerik değilse, sonuna "x" ekle
    if not name[-1].isalnum():
        name = name + "x"
    
    # Eğer ilk karakter alfanumerik değilse, başına "x" ekle
    if not name[0].isalnum():
        name = "x" + name
    
    return name

def create_or_update_collection(
    documents: List[Document],
    embeddings: Embeddings,
    collection_name: str,
    source_name: Optional[str] = None
) -> Chroma:
    """
    Vektör veritabanında bir koleksiyon oluşturur veya günceller.
    
    Args:
        documents: Eklenecek belgeler listesi
        embeddings: Kullanılacak embedding modeli
        collection_name: Koleksiyon adı
        source_name: İsteğe bağlı kaynak adı (filtreleme için)
    
    Returns:
        Güncellenen veya oluşturulan Chroma veritabanı
    """
    # Koleksiyon adını normalize et
    normalized_collection_name = normalize_collection_name(collection_name)
    
    if normalized_collection_name != collection_name:
        print(f"Koleksiyon adı normalize edildi: '{collection_name}' -> '{normalized_collection_name}'")
    
    # Koleksiyon dizinini oluştur
    persist_directory = os.path.join(DB_PATH, normalized_collection_name)
    os.makedirs(persist_directory, exist_ok=True)
    
    try:
        # Koleksiyonu yüklemeyi dene
        vectordb = Chroma(
            collection_name=normalized_collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # Belgeleri ekle
        if source_name:
            # Kaynak filtrelemesi kullanarak mevcut belgeleri sil
            vectordb.delete(
                where={"source": source_name}
            )
        
        # Belgeleri ekle
        vectordb.add_documents(documents)
        vectordb.persist()
        
        return vectordb
        
    except Exception as e:
        # Hata durumunda yeni koleksiyon oluştur
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=normalized_collection_name,
            persist_directory=persist_directory
        )
        vectordb.persist()
        
        return vectordb

def load_collection(
    collection_name: str,
    embeddings: Embeddings
) -> Chroma:
    """
    Var olan bir koleksiyonu yükler.
    Koleksiyon yoksa FileNotFoundError hatası döndürür.
    
    Args:
        collection_name: Yüklenecek koleksiyon adı
        embeddings: Kullanılacak embedding modeli
    
    Returns:
        Chroma veritabanı nesnesi
    """
    # Koleksiyon adını normalize et
    normalized_collection_name = normalize_collection_name(collection_name)
    
    if normalized_collection_name != collection_name:
        print(f"Koleksiyon adı normalize edildi: '{collection_name}' -> '{normalized_collection_name}'")
    
    # Koleksiyon yolunu al
    persist_directory = os.path.join(DB_PATH, normalized_collection_name)
    
    # Koleksiyonun var olup olmadığını kontrol et
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Koleksiyon bulunamadı: {collection_name}")
    
    # Koleksiyonu yükle
    vectordb = Chroma(
        collection_name=normalized_collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    
    return vectordb 