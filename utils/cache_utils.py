"""
DoChat için Cached RAG (Cache-Augmented Retrieval & Generation) modülü.
Bu modül, RAG işlemlerini önbellekleme ile optimize etmek için kullanılır.
"""

import os
import json
import time
import hashlib
import threading
from typing import Dict, List, Tuple, Any, Optional, Union
from functools import lru_cache
import numpy as np
from loguru import logger
from utils.embedding_utils import get_config_param, compute_cosine_similarity

# Önbellek verileri (bellek içinde)
# Thread-safe sözlükler
_embedding_cache = {}  # {text_hash: embedding_vector}
_retrieval_cache = {}  # {query_hash: (query_embedding, [chunk_ids])}
_context_cache = {}    # {query_hash: context_text}
_response_cache = {}   # {query_hash: (response, sources, timestamp)}

# Önbellek kayıt dosya yolları
CACHE_DIR = "db/cache"
EMBEDDING_CACHE_FILE = os.path.join(CACHE_DIR, "embedding_cache.jsonl")
RETRIEVAL_CACHE_FILE = os.path.join(CACHE_DIR, "retrieval_cache.jsonl")
CONTEXT_CACHE_FILE = os.path.join(CACHE_DIR, "context_cache.jsonl")
RESPONSE_CACHE_FILE = os.path.join(CACHE_DIR, "response_cache.jsonl")

# Thread kilitleri
_cache_lock = threading.RLock()

def _ensure_cache_dir():
    """Önbellek dizininin varlığını kontrol eder ve yoksa oluşturur."""
    os.makedirs(CACHE_DIR, exist_ok=True)

def _hash_text(text: str) -> str:
    """Metin için tekil bir hash değeri üretir."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def _save_cache_to_disk():
    """Önbellek verilerini diske kaydeder."""
    try:
        _ensure_cache_dir()
        
        with open(EMBEDDING_CACHE_FILE, 'w', encoding='utf-8') as f:
            for text_hash, embedding in _embedding_cache.items():
                f.write(json.dumps({'hash': text_hash, 'embedding': embedding}, ensure_ascii=False) + '\n')
                
        with open(RETRIEVAL_CACHE_FILE, 'w', encoding='utf-8') as f:
            for query_hash, (query_embedding, chunk_ids) in _retrieval_cache.items():
                f.write(json.dumps({
                    'hash': query_hash, 
                    'embedding': query_embedding,
                    'chunk_ids': chunk_ids
                }, ensure_ascii=False) + '\n')
                
        with open(CONTEXT_CACHE_FILE, 'w', encoding='utf-8') as f:
            for query_hash, context_text in _context_cache.items():
                f.write(json.dumps({
                    'hash': query_hash, 
                    'context': context_text
                }, ensure_ascii=False) + '\n')
                
        with open(RESPONSE_CACHE_FILE, 'w', encoding='utf-8') as f:
            for query_hash, (response, sources, timestamp) in _response_cache.items():
                f.write(json.dumps({
                    'hash': query_hash, 
                    'response': response,
                    'sources': sources,
                    'timestamp': timestamp
                }, ensure_ascii=False) + '\n')
                
        logger.info(f"Önbellek verileri diske kaydedildi. Toplam: {len(_embedding_cache)} embedding, "
                   f"{len(_retrieval_cache)} retrieval, {len(_context_cache)} context, {len(_response_cache)} response.")
    except Exception as e:
        logger.error(f"Önbellek disk kaydı hatası: {e}")

def _load_cache_from_disk():
    """Önbellek verilerini diskten yükler."""
    try:
        _ensure_cache_dir()
        
        if os.path.exists(EMBEDDING_CACHE_FILE):
            with open(EMBEDDING_CACHE_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        _embedding_cache[data['hash']] = data['embedding']
                    except Exception as e:
                        logger.error(f"Embedding önbellek satırı okunamadı: {e}")
                        
        if os.path.exists(RETRIEVAL_CACHE_FILE):
            with open(RETRIEVAL_CACHE_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        _retrieval_cache[data['hash']] = (data['embedding'], data['chunk_ids'])
                    except Exception as e:
                        logger.error(f"Retrieval önbellek satırı okunamadı: {e}")
                        
        if os.path.exists(CONTEXT_CACHE_FILE):
            with open(CONTEXT_CACHE_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        _context_cache[data['hash']] = data['context']
                    except Exception as e:
                        logger.error(f"Context önbellek satırı okunamadı: {e}")
                        
        if os.path.exists(RESPONSE_CACHE_FILE):
            with open(RESPONSE_CACHE_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        _response_cache[data['hash']] = (data['response'], data['sources'], data['timestamp'])
                    except Exception as e:
                        logger.error(f"Response önbellek satırı okunamadı: {e}")
                        
        logger.info(f"Önbellek verileri diskten yüklendi. Toplam: {len(_embedding_cache)} embedding, "
                  f"{len(_retrieval_cache)} retrieval, {len(_context_cache)} context, {len(_response_cache)} response.")
    except Exception as e:
        logger.error(f"Önbellek disk yükleme hatası: {e}")

def _is_cache_enabled() -> bool:
    """Cached RAG'in aktif olup olmadığını kontrol eder."""
    return get_config_param('cached_rag.enabled', True)

def _get_cache_timeout() -> int:
    """Önbellek timeout değerini (saniye) döndürür."""
    return get_config_param('cached_rag.cache_timeout', 86400)  # Varsayılan 24 saat

def _get_similarity_threshold() -> float:
    """Önbellek benzerlik eşik değerini döndürür."""
    return get_config_param('cached_rag.similarity_threshold', 0.92)

def _get_max_cache_items() -> int:
    """Maksimum önbellek öğe sayısını döndürür."""
    return get_config_param('cached_rag.max_cache_items', 10000)

def _is_embedding_cache_enabled() -> bool:
    """Embedding önbelleklemenin aktif olup olmadığını kontrol eder."""
    return get_config_param('cached_rag.cache_levels.embedding', True)

def _is_retrieval_cache_enabled() -> bool:
    """Retrieval önbelleklemenin aktif olup olmadığını kontrol eder."""
    return get_config_param('cached_rag.cache_levels.retrieval', True)

def _is_context_cache_enabled() -> bool:
    """Context önbelleklemenin aktif olup olmadığını kontrol eder."""
    return get_config_param('cached_rag.cache_levels.context', True)

def _is_response_cache_enabled() -> bool:
    """Response önbelleklemenin aktif olup olmadığını kontrol eder."""
    return get_config_param('cached_rag.cache_levels.response', True)

def _is_semantic_cache_enabled() -> bool:
    """Semantic önbelleklemenin aktif olup olmadığını kontrol eder."""
    return get_config_param('cached_rag.semantic_cache', True)

def _manage_cache_size():
    """Önbellek boyutunu yönetir, gerekirse eski öğeleri temizler."""
    max_items = _get_max_cache_items()
    
    with _cache_lock:
        # Embedding cache yönetimi
        if len(_embedding_cache) > max_items:
            # Fazla öğeleri kaldır - rastgele temizleme
            items = list(_embedding_cache.items())
            # Rastgele %20'sini temizle
            num_to_remove = int(max_items * 0.2)
            for item in items[:num_to_remove]:
                del _embedding_cache[item[0]]
                
        # Response cache yönetimi - zaman damgalarına göre eski olanları temizle
        if len(_response_cache) > max_items:
            current_time = time.time()
            timeout = _get_cache_timeout()
            
            # Zaman aşımına uğramış öğeleri temizle
            expired_keys = []
            for query_hash, (_, _, timestamp) in _response_cache.items():
                if current_time - timestamp > timeout:
                    expired_keys.append(query_hash)
            
            for key in expired_keys:
                del _response_cache[key]
            
            # Hala fazla öğe varsa, en eski olanları temizle
            if len(_response_cache) > max_items:
                # Zaman damgalarına göre sırala
                sorted_items = sorted(
                    _response_cache.items(), 
                    key=lambda x: x[1][2]  # timestamp'e göre sırala
                )
                
                # En eski %20'yi temizle
                num_to_remove = int(max_items * 0.2)
                for i in range(num_to_remove):
                    if i < len(sorted_items):
                        del _response_cache[sorted_items[i][0]]

def _periodic_save_cache():
    """Periyodik olarak önbelleği diske kaydet."""
    _save_cache_to_disk()
    # 30 dakikada bir tekrarla
    threading.Timer(1800, _periodic_save_cache).start()

def init_cache():
    """Önbelleği başlatır, disk verilerini yükler."""
    with _cache_lock:
        _load_cache_from_disk()
        # Periyodik kaydetme işlemini başlat
        threading.Timer(1800, _periodic_save_cache).start()
    logger.info("Cached RAG sistemi başlatıldı.")

def clear_cache():
    """Tüm önbellek verilerini temizler."""
    with _cache_lock:
        _embedding_cache.clear()
        _retrieval_cache.clear()
        _context_cache.clear()
        _response_cache.clear()
        _save_cache_to_disk()
    logger.info("Önbellek tamamen temizlendi.")

@lru_cache(maxsize=1024)
def get_cached_embedding(text: str) -> Optional[List[float]]:
    """
    Metne ait daha önce hesaplanmış embedding'i döndürür.
    Önbellekte yoksa None döner.
    
    Args:
        text: Embedding'i aranacak metin
        
    Returns:
        Embedding vektörü veya None
    """
    if not _is_cache_enabled() or not _is_embedding_cache_enabled():
        return None
        
    text_hash = _hash_text(text)
    with _cache_lock:
        return _embedding_cache.get(text_hash)

def cache_embedding(text: str, embedding: List[float]) -> None:
    """
    Metin ve embedding'i önbelleğe kaydeder.
    
    Args:
        text: Önbelleğe eklenecek metin
        embedding: Metin için hesaplanmış embedding vektörü
    """
    if not _is_cache_enabled() or not _is_embedding_cache_enabled():
        return
        
    text_hash = _hash_text(text)
    with _cache_lock:
        _embedding_cache[text_hash] = embedding
        _manage_cache_size()

def get_cached_retrieval(query: str, query_embedding: List[float]) -> Optional[List[str]]:
    """
    Sorguya ait önbellekteki retrieval sonuçlarını döndürür.
    Önbellekte yoksa veya semantic cache aktif değilse None döner.
    
    Args:
        query: Sorgu metni
        query_embedding: Sorgu için hesaplanmış embedding
        
    Returns:
        Chunk ID'leri listesi veya None
    """
    if not _is_cache_enabled() or not _is_retrieval_cache_enabled():
        return None
        
    # Doğrudan anahtar eşleşmesini kontrol et
    query_hash = _hash_text(query)
    with _cache_lock:
        exact_match = _retrieval_cache.get(query_hash)
        if exact_match:
            logger.info(f"Retrieval önbellekte tam eşleşme bulundu: {query}")
            return exact_match[1]  # chunk_ids
    
    # Semantic cache aktif değilse çık
    if not _is_semantic_cache_enabled():
        return None
    
    # Benzerlik tabanlı arama
    similarity_threshold = _get_similarity_threshold()
    best_match = None
    best_similarity = 0.0
    
    with _cache_lock:
        for cached_hash, (cached_embedding, chunk_ids) in _retrieval_cache.items():
            similarity = compute_cosine_similarity(query_embedding, cached_embedding)
            if similarity > similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = chunk_ids
    
    if best_match:
        logger.info(f"Retrieval önbellekte anlamsal eşleşme bulundu. Benzerlik: {best_similarity:.4f}")
    
    return best_match

def cache_retrieval(query: str, query_embedding: List[float], chunk_ids: List[str]) -> None:
    """
    Sorgu, embedding ve retrieval sonuçlarını önbelleğe kaydeder.
    
    Args:
        query: Sorgu metni
        query_embedding: Sorgu embedding'i
        chunk_ids: Sorguya ait retrieval sonuçları (chunk ID'leri)
    """
    if not _is_cache_enabled() or not _is_retrieval_cache_enabled():
        return
        
    query_hash = _hash_text(query)
    with _cache_lock:
        _retrieval_cache[query_hash] = (query_embedding, chunk_ids)
        _manage_cache_size()

def get_cached_context(query: str, query_embedding: List[float]) -> Optional[str]:
    """
    Sorguya ait önbellekteki context'i döndürür.
    Önbellekte yoksa veya semantic cache aktif değilse None döner.
    
    Args:
        query: Sorgu metni
        query_embedding: Sorgu için hesaplanmış embedding
        
    Returns:
        Context metni veya None
    """
    if not _is_cache_enabled() or not _is_context_cache_enabled():
        return None
        
    # Doğrudan anahtar eşleşmesini kontrol et
    query_hash = _hash_text(query)
    with _cache_lock:
        exact_match = _context_cache.get(query_hash)
        if exact_match:
            logger.info(f"Context önbellekte tam eşleşme bulundu: {query}")
            return exact_match
    
    # Semantic cache aktif değilse çık
    if not _is_semantic_cache_enabled():
        return None
    
    # TODO: Context için semantic cache yapılabilir, ancak şu an için gerekli değil
    return None

def cache_context(query: str, context: str) -> None:
    """
    Sorgu ve context'i önbelleğe kaydeder.
    
    Args:
        query: Sorgu metni
        context: Sorgu için hazırlanmış context
    """
    if not _is_cache_enabled() or not _is_context_cache_enabled():
        return
        
    query_hash = _hash_text(query)
    with _cache_lock:
        _context_cache[query_hash] = context
        _manage_cache_size()

def get_cached_response(query: str, query_embedding: List[float]) -> Optional[Tuple[str, List[str]]]:
    """
    Sorguya ait önbellekteki yanıtı döndürür.
    Önbellekte yoksa, zaman aşımına uğramışsa veya semantic cache aktif değilse None döner.
    
    Args:
        query: Sorgu metni
        query_embedding: Sorgu için hesaplanmış embedding
        
    Returns:
        (yanıt, kaynaklar) tuple'ı veya None
    """
    if not _is_cache_enabled() or not _is_response_cache_enabled():
        return None
        
    current_time = time.time()
    timeout = _get_cache_timeout()
    
    # Doğrudan anahtar eşleşmesini kontrol et
    query_hash = _hash_text(query)
    with _cache_lock:
        exact_match = _response_cache.get(query_hash)
        if exact_match:
            response, sources, timestamp = exact_match
            # Zaman aşımı kontrolü
            if current_time - timestamp <= timeout:
                logger.info(f"Response önbellekte tam eşleşme bulundu: {query}")
                return response, sources
            else:
                # Zaman aşımına uğramış, önbellekten çıkar
                del _response_cache[query_hash]
    
    # Semantic cache aktif değilse çık
    if not _is_semantic_cache_enabled():
        return None
    
    # Benzerlik tabanlı arama
    similarity_threshold = _get_similarity_threshold()
    best_match = None
    best_similarity = 0.0
    
    with _cache_lock:
        for cached_hash, (response, sources, timestamp) in _response_cache.items():
            # Zaman aşımı kontrolü
            if current_time - timestamp > timeout:
                continue
                
            # Önceden hesaplanmış query embedding'lerini retrieval cache'den al
            if cached_hash in _retrieval_cache:
                cached_embedding = _retrieval_cache[cached_hash][0]
                similarity = compute_cosine_similarity(query_embedding, cached_embedding)
                if similarity > similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (response, sources)
    
    if best_match:
        logger.info(f"Response önbellekte anlamsal eşleşme bulundu. Benzerlik: {best_similarity:.4f}")
    
    return best_match

def cache_response(query: str, response: str, sources: List[str]) -> None:
    """
    Sorgu, yanıt ve kaynakları önbelleğe kaydeder.
    
    Args:
        query: Sorgu metni
        response: Yanıt metni
        sources: Kaynak listesi
    """
    if not _is_cache_enabled() or not _is_response_cache_enabled():
        return
        
    query_hash = _hash_text(query)
    current_time = time.time()
    
    with _cache_lock:
        _response_cache[query_hash] = (response, sources, current_time)
        _manage_cache_size()

def get_cache_stats() -> Dict[str, Any]:
    """
    Önbellek istatistiklerini döndürür.
    
    Returns:
        İstatistikler içeren sözlük
    """
    with _cache_lock:
        stats = {
            "embedding_cache_size": len(_embedding_cache),
            "retrieval_cache_size": len(_retrieval_cache),
            "context_cache_size": len(_context_cache),
            "response_cache_size": len(_response_cache),
            "cache_enabled": _is_cache_enabled(),
            "semantic_cache_enabled": _is_semantic_cache_enabled(),
            "similarity_threshold": _get_similarity_threshold(),
            "cache_timeout": _get_cache_timeout()
        }
    return stats

# Başlangıçta önbelleği yükle
try:
    init_cache()
except Exception as e:
    logger.error(f"Önbellek başlatma hatası: {e}") 