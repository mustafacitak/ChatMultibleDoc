"""
DoChat için Binary Quantization modülü.
Bu modül, embedding vektörlerini 1-bit temsillere (binary) dönüştürerek
bellek kullanımını azaltıp, benzerlik hesaplamalarını hızlandırır.
"""

import numpy as np
from loguru import logger
from utils.embedding_utils import get_config_param
from typing import List, Dict, Any, Tuple, Union, Optional
import functools

def binarize_vector(vector: List[float], threshold: Optional[float] = None) -> np.ndarray:
    """
    Embedding vektörünü binary (1-bit) temsile dönüştürür.
    
    Args:
        vector: Orijinal embedding vektörü
        threshold: Eşik değeri (None ise medyan değeri kullanılır)
        
    Returns:
        np.ndarray: Binary vektör (0 ve 1'lerden oluşan)
    """
    # NumPy array'e dönüştür
    arr = np.array(vector)
    
    # Eşik değeri belirtilmemişse medyan kullan
    if threshold is None:
        threshold = np.median(arr)
    
    # Eşik değerine göre ikili temsile dönüştür (0/1)
    binary_vector = (arr >= threshold).astype(np.int8)
    
    return binary_vector

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """
    İki binary vektör arasındaki Hamming mesafesini hesaplar
    (farklı bitlerin sayısı).
    
    Args:
        a: İlk binary vektör
        b: İkinci binary vektör
        
    Returns:
        int: Hamming mesafesi
    """
    # XOR işlemiyle farklı bitleri bul ve topla
    return np.sum(a != b)

def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    İki binary vektör arasındaki benzerliği hesaplar (0-1 arası).
    1 - (hamming_distance / vektör_boyutu) formülüyle hesaplanır.
    
    Args:
        a: İlk binary vektör
        b: İkinci binary vektör
        
    Returns:
        float: Benzerlik skoru (0-1 arası)
    """
    distance = hamming_distance(a, b)
    return 1.0 - (distance / len(a))

def pack_binary_vector(binary_vector: np.ndarray) -> np.ndarray:
    """
    Binary vektörü bit düzeyinde sıkıştırır (8 kat tasarruf).
    
    Args:
        binary_vector: 0 ve 1'lerden oluşan binary vektör
        
    Returns:
        np.ndarray: Sıkıştırılmış binary vektör
    """
    # Vektör boyutunu 8'in katına tamamla
    padding = 8 - (len(binary_vector) % 8) if len(binary_vector) % 8 != 0 else 0
    if padding > 0:
        padded_vector = np.pad(binary_vector, (0, padding), 'constant', constant_values=0)
    else:
        padded_vector = binary_vector
    
    # 8 biti tek byte'a sıkıştır
    packed_vector = np.packbits(padded_vector)
    
    return packed_vector

def unpack_binary_vector(packed_vector: np.ndarray, original_length: int) -> np.ndarray:
    """
    Sıkıştırılmış binary vektörü orijinal formuna geri döndürür.
    
    Args:
        packed_vector: Sıkıştırılmış binary vektör
        original_length: Orijinal vektörün uzunluğu
        
    Returns:
        np.ndarray: Orijinal binary vektör
    """
    # Byte'ları bitlere aç
    unpacked_vector = np.unpackbits(packed_vector)
    
    # Orijinal boyuta kes
    return unpacked_vector[:original_length]

class BinaryQuantizer:
    def __init__(self):
        """
        Binary Quantizer sınıfını başlatır. Bu sınıf, vektörlerin binary temsillerle
        işlenmesine olanak tanır.
        """
        # Konfigürasyonu yükle
        self.enabled = get_config_param('binary_quantization.enabled', True)
        self.use_packing = get_config_param('binary_quantization.use_packing', True)
        self.hybrid_ratio = get_config_param('binary_quantization.hybrid_ratio', 0.0)
        
        # Önbellek
        self._vector_cache = {}
    
    def quantize(self, vector: List[float]) -> Dict[str, Any]:
        """
        Vektörü binary şekilde kuantalama işlemi yapar.
        
        Args:
            vector: Kuantalanacak embedding vektörü
            
        Returns:
            Dict[str, Any]: Kuantalama sonuçları (binary vektör ve meta bilgiler)
        """
        if not self.enabled:
            return {'original': vector, 'binary': None, 'is_quantized': False}
        
        vector_key = str(hash(tuple(vector)))
        
        # Önbellekte var mı kontrol et
        if vector_key in self._vector_cache:
            return self._vector_cache[vector_key]
        
        # Binary vektöre dönüştür
        binary_vector = binarize_vector(vector)
        
        # Hybrid mod için orijinal vektörü sakla
        result = {
            'original': vector if self.hybrid_ratio > 0 else None,
            'binary': pack_binary_vector(binary_vector) if self.use_packing else binary_vector,
            'is_quantized': True,
            'is_packed': self.use_packing,
            'original_length': len(vector),
            'hybrid_ratio': self.hybrid_ratio
        }
        
        # Önbelleğe ekle
        self._vector_cache[vector_key] = result
        
        return result
    
    def compute_similarity(self, query_vec: Union[List[float], Dict[str, Any]], 
                           doc_vec: Union[List[float], Dict[str, Any]]) -> float:
        """
        İki vektör arasındaki benzerliği hesaplar. Vektörler orijinal veya kuantalanmış olabilir.
        
        Args:
            query_vec: Sorgu vektörü veya kuantalama sonucu
            doc_vec: Doküman vektörü veya kuantalama sonucu
            
        Returns:
            float: Benzerlik skoru (0-1 arası)
        """
        if not self.enabled:
            # Binary quantization devre dışıysa standart kosinüs benzerliği kullan
            from utils.embedding_utils import compute_cosine_similarity
            return compute_cosine_similarity(
                query_vec if isinstance(query_vec, list) else query_vec['original'],
                doc_vec if isinstance(doc_vec, list) else doc_vec['original'],
                allow_binary=False
            )
        
        # Vektörleri kuantalama formatına dönüştür
        query_quantized = query_vec if isinstance(query_vec, dict) else self.quantize(query_vec)
        doc_quantized = doc_vec if isinstance(doc_vec, dict) else self.quantize(doc_vec)
        
        # Binary vektörleri çıkar
        query_binary = query_quantized['binary']
        doc_binary = doc_quantized['binary']
        
        # Packed formatındaysa unpack et
        if query_quantized.get('is_packed', False):
            query_binary = unpack_binary_vector(query_binary, query_quantized['original_length'])
        if doc_quantized.get('is_packed', False):
            doc_binary = unpack_binary_vector(doc_binary, doc_quantized['original_length'])
        
        # Binary benzerlik hesapla
        binary_similarity = hamming_similarity(query_binary, doc_binary)
        
        # Hybrid mod aktif mi?
        hybrid_ratio = max(query_quantized.get('hybrid_ratio', 0.0), doc_quantized.get('hybrid_ratio', 0.0))
        
        if hybrid_ratio > 0 and query_quantized.get('original') is not None and doc_quantized.get('original') is not None:
            # Orijinal vektörlerle kosinüs benzerliği hesapla
            from utils.embedding_utils import compute_cosine_similarity
            cosine_similarity = compute_cosine_similarity(query_quantized['original'], doc_quantized['original'], allow_binary=False)
            
            # Ağırlıklı ortalama al
            return (binary_similarity * (1 - hybrid_ratio)) + (cosine_similarity * hybrid_ratio)
        
        return binary_similarity
    
    def batch_quantize(self, vectors: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Bir vektör listesini toplu şekilde kuantalar.
        
        Args:
            vectors: Kuantalanacak vektörler listesi
            
        Returns:
            List[Dict[str, Any]]: Kuantalama sonuçları listesi
        """
        return [self.quantize(vec) for vec in vectors]
    
    def batch_compute_similarity(self, query_vec: Union[List[float], Dict[str, Any]], 
                                doc_vecs: List[Union[List[float], Dict[str, Any]]]) -> List[float]:
        """
        Sorgu vektörü ile birden fazla doküman vektörü arasındaki benzerlikleri hesaplar.
        
        Args:
            query_vec: Sorgu vektörü veya kuantalama sonucu
            doc_vecs: Doküman vektörleri veya kuantalama sonuçları listesi
            
        Returns:
            List[float]: Benzerlik skorları listesi
        """
        return [self.compute_similarity(query_vec, doc_vec) for doc_vec in doc_vecs]
    
    def clear_cache(self):
        """
        Vektör önbelleğini temizler.
        """
        self._vector_cache.clear()

# Singleton instance
_binary_quantizer = None

def get_binary_quantizer() -> BinaryQuantizer:
    """
    Binary Quantizer singleton instance'ını döndürür.
    
    Returns:
        BinaryQuantizer: Binary Quantizer örneği
    """
    global _binary_quantizer
    if _binary_quantizer is None:
        _binary_quantizer = BinaryQuantizer()
    return _binary_quantizer 