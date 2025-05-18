"""
DoChat için Adaptive-RAG (Adaptive Retrieval-Augmented Generation) modülü.
Bu modül, sorgu karmaşıklığına göre retrieval stratejisini dinamik olarak ayarlar.
"""

from loguru import logger
import re
from utils.embedding_utils import get_config_param
from typing import Dict, Any, List, Tuple, Optional

def analyze_query_complexity(query: str) -> Dict[str, Any]:
    """
    Sorgunun karmaşıklığını analiz eder ve kompleksite skorunu hesaplar.
    Karmaşıklık faktörleri:
    1. Uzunluk (kelime sayısı)
    2. Soru kelimesi sayısı ve tipi
    3. Teknik terim/özel isim varlığı
    4. Koşul/kısıtlama içerip içermediği
    
    Args:
        query: Sorgu metni
        
    Returns:
        Dict: Karmaşıklık analizi sonuçları
    """
    query = query.strip()
    
    # Kelime sayısı
    words = query.split()
    word_count = len(words)
    
    # Soru kelimeleri sayısı
    question_words = ["ne", "neden", "nasıl", "nerede", "ne zaman", "kim", "hangisi", "kaç", "niçin"]
    question_word_count = sum(1 for word in words if word.lower() in question_words)
    
    # Karşılaştırma ifadeleri
    comparison_phrases = ["arasındaki fark", "karşılaştır", "kıyasla", "daha iyi", "benzerlik", "versus", "vs"]
    has_comparison = any(phrase in query.lower() for phrase in comparison_phrases)
    
    # Koşullu ifadeler
    conditional_phrases = ["eğer", "ancak", "fakat", "ama", "veya", "ya da", "yahut", "ve", "ile"]
    conditional_count = sum(1 for word in words if word.lower() in conditional_phrases)
    
    # Teknik terim varlığı (bu, domain'e bağlı olabilir, örnek olarak genel bir yaklaşım)
    # Büyük harfle başlayan kelimeleri teknik terim veya özel isim olarak kabul ediyoruz
    capitalized_terms = sum(1 for word in words if word and word[0].isupper())
    
    # Karmaşıklık skoru hesaplama (0-100 arası)
    complexity_score = 0
    complexity_score += min(40, word_count * 2)  # Uzunluk: maksimum 40 puan
    complexity_score += min(20, question_word_count * 5)  # Soru kelimeleri: maksimum 20 puan
    complexity_score += 15 if has_comparison else 0  # Karşılaştırma: 15 puan
    complexity_score += min(10, conditional_count * 2)  # Koşullu ifadeler: maksimum 10 puan
    complexity_score += min(15, capitalized_terms * 3)  # Teknik terimler: maksimum 15 puan
    
    # Skor kategorisini belirleme
    complexity_category = "basit"
    if complexity_score > 70:
        complexity_category = "karmaşık" 
    elif complexity_score > 40:
        complexity_category = "orta"
    
    return {
        "score": complexity_score,
        "category": complexity_category,
        "word_count": word_count,
        "question_word_count": question_word_count,
        "has_comparison": has_comparison,
        "conditional_count": conditional_count,
        "capitalized_terms": capitalized_terms
    }

def get_adaptive_chunk_count(query: str) -> int:
    """
    Sorgu karmaşıklığına göre uygun chunk sayısını belirler.
    
    Args:
        query: Sorgu metni
        
    Returns:
        int: Kullanılacak chunk sayısı (top_k değeri)
    """
    # Konfigürasyondan varsayılan değerleri al
    config_base_top_k = get_config_param('chunking.top_k', 5)
    use_adaptive_rag = get_config_param('adaptive_rag.enabled', True)
    
    if not use_adaptive_rag:
        return config_base_top_k
    
    # Adaptive RAG konfigürasyonunu al
    simple_query_top_k = get_config_param('adaptive_rag.simple_query_top_k', 3)
    medium_query_top_k = get_config_param('adaptive_rag.medium_query_top_k', 5)
    complex_query_top_k = get_config_param('adaptive_rag.complex_query_top_k', 8)
    
    # Sorgu karmaşıklığını analiz et
    complexity = analyze_query_complexity(query)
    category = complexity["category"]
    
    # Karmaşıklığa göre chunk sayısı belirle
    if category == "basit":
        chunk_count = simple_query_top_k
    elif category == "orta":
        chunk_count = medium_query_top_k
    else:  # "karmaşık"
        chunk_count = complex_query_top_k
    
    logger.info(f"Adaptive-RAG: Sorgu karmaşıklığı \"{category}\" (skor: {complexity['score']}). Kullanılacak chunk sayısı: {chunk_count}")
    return chunk_count

def get_adaptive_context_size(query: str, default_size: int = 15000) -> int:
    """
    Sorgu karmaşıklığına göre uygun bağlam boyutunu belirler.
    
    Args:
        query: Sorgu metni
        default_size: Varsayılan bağlam boyutu
        
    Returns:
        int: Kullanılacak maksimum bağlam boyutu (karakter sayısı)
    """
    use_adaptive_rag = get_config_param('adaptive_rag.enabled', True)
    
    if not use_adaptive_rag:
        return default_size
    
    # Adaptive RAG konfigürasyonunu al
    adaptive_context_sizing = get_config_param('adaptive_rag.adaptive_context_sizing', True)
    
    if not adaptive_context_sizing:
        return default_size
    
    # Farklı karmaşıklık seviyeleri için bağlam boyutları
    simple_context_size = get_config_param('adaptive_rag.simple_context_size', 7500)
    medium_context_size = get_config_param('adaptive_rag.medium_context_size', 15000)
    complex_context_size = get_config_param('adaptive_rag.complex_context_size', 20000)
    
    # Sorgu karmaşıklığını analiz et
    complexity = analyze_query_complexity(query)
    category = complexity["category"]
    
    # Karmaşıklığa göre bağlam boyutu belirle
    if category == "basit":
        context_size = simple_context_size
    elif category == "orta":
        context_size = medium_context_size
    else:  # "karmaşık"
        context_size = complex_context_size
    
    logger.info(f"Adaptive-RAG: Sorgu karmaşıklığı \"{category}\" için maksimum bağlam boyutu: {context_size} karakter")
    return context_size 