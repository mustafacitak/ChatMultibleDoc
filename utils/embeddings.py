import os
import time
from typing import List, Optional, Dict, Any
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.embeddings import Embeddings

# CacheBackedEmbeddings değişmiş olabilir, o yüzden şimdi kendi basit önbelleğimizi kullanacağız
class SimpleCachedEmbeddings:
    """
    Basit bir önbellekli embedding sarmalayıcısı.
    """
    def __init__(self, underlying_embedder: Embeddings, batch_size: int = 5):
        self.embedder = underlying_embedder
        self.cache = {}  # Metin -> embedding vektörü eşleştirmesi
        self.batch_size = batch_size  # API rate limit aşımını önlemek için küçük batch'ler
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Belgeleri gömme vektörlerine dönüştürür ve önbellekler."""
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Önbellekte olmayanları bul
        for i, text in enumerate(texts):
            if text in self.cache:
                results.append(self.cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Uncached tekstleri batch'ler halinde işle
        if uncached_texts:
            for i in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[i:i+self.batch_size]
                try:
                    batch_embeddings = self.embedder.embed_documents(batch)
                    
                    # Embeddings'leri cache'e ekle
                    for j, embedding in enumerate(batch_embeddings):
                        text = uncached_texts[i+j]
                        self.cache[text] = embedding
                    
                    # API rate limit'i aşmamak için kısa bir bekleme
                    if i + self.batch_size < len(uncached_texts):
                        time.sleep(0.5)  # 500 ms bekle
                        
                except Exception as e:
                    # Hata durumunda batch boyutunu azalt ve tekrar dene
                    if self.batch_size > 1 and "quota" in str(e).lower():
                        self.batch_size = max(1, self.batch_size // 2)
                        print(f"API limit aşıldı, batch boyutu {self.batch_size}'e düşürüldü. Tekrar deneniyor...")
                        time.sleep(1)  # 1 saniye bekle
                        # Tekrar aynı batch ile dene
                        batch_embeddings = self.embedder.embed_documents(batch)
                        
                        # Embeddings'leri cache'e ekle
                        for j, embedding in enumerate(batch_embeddings):
                            text = uncached_texts[i+j]
                            self.cache[text] = embedding
                    else:
                        raise e
        
        # Son sonuçları düzenle
        final_results = [None] * len(texts)
        cached_count = 0
        for i, text in enumerate(texts):
            final_results[i] = self.cache[text]
            
        return final_results
    
    def embed_query(self, text: str) -> List[float]:
        """Bir sorguyu gömme vektörüne dönüştürür ve önbellekler."""
        if text in self.cache:
            return self.cache[text]
        else:
            embedding = self.embedder.embed_query(text)
            self.cache[text] = embedding
            return embedding

# InMemoryByteStore artık langchain.cache'de bulunmadığı için basit bir uyumluluk sınıfı
class InMemoryByteStore:
    """
    Basit bir bellek-içi saklama sınıfı.
    """
    def __init__(self):
        self.store = {}

    def get(self, key: str) -> Optional[bytes]:
        return self.store.get(key)

    def set(self, key: str, value: bytes) -> None:
        self.store[key] = value

    def delete(self, key: str) -> None:
        if key in self.store:
            del self.store[key]

    def yield_keys(self, prefix: str = "") -> List[str]:
        return [k for k in self.store.keys() if k.startswith(prefix)]

def get_google_embeddings() -> Embeddings:
    """
    Eski Google Embeddings API kullanımını korumak için.
    Şimdi bu fonksiyon yerel embeddings modelini kullanıyor.
    
    Bu fonksiyonu kullanmayın, bunun yerine get_local_embeddings() kullanın.
    """
    print("UYARI: get_google_embeddings() artık kullanımdan kaldırıldı. Yerine get_local_embeddings() kullanılıyor.")
    return get_local_embeddings()

def get_local_embeddings() -> Embeddings:
    """
    Yerel çalışan SentenceTransformer temelli embedding modeli döndürür.
    Google API anahtarı gerekli değildir.
    """
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    print(f"SentenceTransformer embedding modeli yükleniyor: {model_name}")
    return SentenceTransformerEmbeddings(model_name=model_name)

def get_cached_embeddings(embedding_model: Embeddings) -> SimpleCachedEmbeddings:
    """
    Aynı metinler için tekrar embedding hesaplamayı önleyen,
    önbellekli bir embedding sarmalayıcısı döndürür.
    """
    # Basit önbellekli embedding sarmalayıcısını döndür
    return SimpleCachedEmbeddings(underlying_embedder=embedding_model, batch_size=5) 