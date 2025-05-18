"""
DoChat için FUNNELRAG (İki Aşamalı Retrieval) modülü.
Bu modül, önce hafif bir filtreleme ile geniş bir aday kümesi oluşturur, 
sonra bu adaylar üzerinde daha hassas bir embedding karşılaştırması yapar.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
from utils.embedding_utils import get_config_param, compute_cosine_similarity, generate_embedding
from typing import List, Dict, Any, Tuple, Optional
import re

class FunnelRAG:
    def __init__(self):
        """
        FunnelRAG sınıfını başlatır. Bu sınıf, hafif ve ağır iki aşamalı retrieval işlemini yönetir.
        """
        # Konfigürasyonu yükle
        self.enabled = get_config_param('funnel_rag.enabled', True)
        self.coarse_candidate_count = get_config_param('funnel_rag.coarse_candidate_count', 100)
        self.use_tfidf = get_config_param('funnel_rag.use_tfidf', True)
        self.use_keyword = get_config_param('funnel_rag.use_keyword', True)
        self.fine_candidate_ratio = get_config_param('funnel_rag.fine_candidate_ratio', 0.3)
        
        # TF-IDF vektörleyici
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunks_for_tfidf = []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Metin içindeki önemli anahtar kelimeleri çıkarır.
        
        Args:
            text: İşlenecek metin
            
        Returns:
            List[str]: Anahtar kelimeler listesi
        """
        # Noktalama işaretlerini ve gereksiz kelimeleri kaldır
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Stopwords (Türkçe)
        stopwords = ["ve", "veya", "ile", "bu", "şu", "o", "bir", "için", "mi", "mı", "ne", "nasıl",
                    "de", "da", "ki", "ama", "fakat", "lakin", "ancak", "ise", "ya", "dahi"]
        
        # Stopwords olmayan ve minimum 3 karakter olan kelimeleri filtrele
        keywords = [word for word in words if word not in stopwords and len(word) >= 3]
        
        return keywords
    
    def _calculate_keyword_similarity(self, query: str, chunks: List[Dict[str, Any]]) -> List[float]:
        """
        Sorgu ve her chunk arasındaki anahtar kelime benzerliğini hesaplar.
        
        Args:
            query: Sorgu metni
            chunks: Chunk'lar listesi
            
        Returns:
            List[float]: Her chunk için benzerlik skorları
        """
        query_keywords = set(self._extract_keywords(query))
        
        if not query_keywords:
            # Anahtar kelime çıkarılamadıysa tümünü eşit kabul et
            return [1.0] * len(chunks)
        
        similarities = []
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            chunk_keywords = set(self._extract_keywords(chunk_text))
            
            # Jaccard benzerliği (kesişim/birleşim)
            if not chunk_keywords:
                similarities.append(0.0)
            else:
                intersection = len(query_keywords.intersection(chunk_keywords))
                union = len(query_keywords.union(chunk_keywords))
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        return similarities
    
    def _build_tfidf_index(self, chunks: List[Dict[str, Any]]):
        """
        Chunk'lar için TF-IDF indeksi oluşturur.
        
        Args:
            chunks: İndekslenecek chunk'lar
        """
        # Chunk metinlerini çıkar
        texts = [chunk.get('text', '') for chunk in chunks]
        self.chunks_for_tfidf = chunks
        
        # TF-IDF vektörleyici oluştur ve eğit
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True, 
            analyzer='word',
            min_df=2,  # Minimum 2 dokümanda geçen terimler
            max_df=0.95  # En fazla %95 dokümanda geçen terimler
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF indeksi oluşturuldu: {len(texts)} dokuman, {self.tfidf_matrix.shape[1]} terim")
    
    def _calculate_tfidf_similarity(self, query: str) -> List[float]:
        """
        Sorgu ve TF-IDF matrisindeki her vektör arasındaki benzerliği hesaplar.
        
        Args:
            query: Sorgu metni
            
        Returns:
            List[float]: Her chunk için TF-IDF benzerlik skorları
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            logger.warning("TF-IDF indeksi bulunamadı, sıfır benzerlik döndürülüyor.")
            return [0.0] * len(self.chunks_for_tfidf)
        
        # Sorgu vektörünü oluştur
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Her chunk ile benzerliğini hesapla (Kosinüs benzerliği)
        similarities = (query_vector * self.tfidf_matrix.T).toarray()[0]
        
        return similarities
    
    def get_top_k_chunks(self, query: str, chunks: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """
        FunnelRAG yaklaşımıyla sorguya en alakalı top-k chunk'ları getirir.
        
        Args:
            query: Sorgu metni
            chunks: Tüm chunk'ları içeren liste
            k: Döndürülecek maksimum chunk sayısı
            
        Returns:
            List[Dict[str, Any]]: En alakalı k chunk
        """
        if not self.enabled or len(chunks) <= k:
            logger.info(f"FUNNELRAG: Standart retrieval kullanılıyor (FUNNELRAG devre dışı veya az sayıda chunk var)")
            return chunks[:k]
        
        # Coarse kandidat sayısını belirleme
        coarse_k = min(self.coarse_candidate_count, len(chunks))
        
        # TF-IDF indeksi oluştur (ilk çağrıda)
        if self.use_tfidf and (self.tfidf_vectorizer is None or len(self.chunks_for_tfidf) != len(chunks)):
            self._build_tfidf_index(chunks)
        
        # 1. AŞAMA: Hafif filtreleme (Coarse-grained selection)
        coarse_candidates = []
        coarse_scores = []
        
        if self.use_tfidf:
            # TF-IDF benzerliklerini hesapla
            tfidf_scores = self._calculate_tfidf_similarity(query)
            coarse_scores.extend(tfidf_scores)
            
        if self.use_keyword:
            # Anahtar kelime benzerliklerini hesapla
            keyword_scores = self._calculate_keyword_similarity(query, chunks)
            
            # Skorları birleştir (TF-IDF varsa ortalama al, yoksa direkt kullan)
            if self.use_tfidf:
                for i in range(len(coarse_scores)):
                    coarse_scores[i] = (coarse_scores[i] + keyword_scores[i]) / 2
            else:
                coarse_scores = keyword_scores
        
        # Skorlara göre indeksleri sırala
        coarse_indices = np.argsort(coarse_scores)[::-1][:coarse_k]
        
        # Seçilen indekslerdeki chunk'ları al
        coarse_candidates = [chunks[i] for i in coarse_indices]
        
        logger.info(f"FUNNELRAG: Coarse aşaması tamamlandı. {len(chunks)} chunk'tan {len(coarse_candidates)} aday seçildi.")
        
        # 2. AŞAMA: Tam embedding karşılaştırması (Fine-grained selection)
        # Sorgu embedding'ini oluştur
        query_emb = generate_embedding(query)
        
        # Her adayın embedding'i mevcut mu kontrol et, yoksa oluştur
        for candidate in coarse_candidates:
            if 'embedding' not in candidate:
                candidate['embedding'] = generate_embedding(candidate.get('text', ''))
        
        # Tam embedding benzerliklerini hesapla
        similarities = []
        for candidate in coarse_candidates:
            sim = compute_cosine_similarity(query_emb, candidate['embedding'])
            similarities.append(sim)
            candidate['similarity'] = sim
        
        # Benzerliğe göre sırala ve top-k'yı döndür
        sorted_candidates = sorted(coarse_candidates, key=lambda x: x.get('similarity', 0), reverse=True)
        fine_candidates = sorted_candidates[:k]
        
        logger.info(f"FUNNELRAG: Fine aşaması tamamlandı. {len(coarse_candidates)} adaydan {len(fine_candidates)} chunk seçildi.")
        
        return fine_candidates 