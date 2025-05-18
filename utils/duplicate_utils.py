import yaml
import asyncio
import functools
from loguru import logger
from utils.embedding_utils import compute_cosine_similarity

# Hata yakalama dekoratörü
def duplicate_error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Duplikasyon hatası - Fonksiyon: {func.__name__}, Hata: {e}")
            return False
    return wrapper

# Async hata yakalama dekoratörü
def async_duplicate_error_handler(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Async duplikasyon hatası - Fonksiyon: {func.__name__}, Hata: {e}")
            return False
    return wrapper

@duplicate_error_handler
def is_duplicate_by_embedding(new_emb, existing_embs, threshold=0.80):
    """
    Embedding vektörleri arasındaki benzerliği ölçerek duplikasyon kontrolü yapar
    
    Args:
        new_emb: Yeni eklenen içeriğin embedding vektörü
        existing_embs: Mevcut embedding vektörleri listesi
        threshold: Benzerlik eşik değeri (0.0-1.0)
        
    Returns:
        bool: Duplikasyon varsa True, yoksa False
    """
    if not existing_embs:
        return False
        
    logger.info(f"Embedding tabanlı duplikasyon kontrolü: {len(existing_embs)} vektör ile karşılaştırılıyor")
    sims = [compute_cosine_similarity(new_emb, emb) for emb in existing_embs]
    max_sim = max(sims) if sims else 0
    
    logger.info(f"En yüksek benzerlik skoru: {max_sim:.4f} (Eşik: {threshold:.4f})")
    return max_sim > threshold

@async_duplicate_error_handler
async def is_duplicate_by_embedding_async(new_emb, existing_embs, threshold=0.80):
    """
    Async versiyonu - Embedding vektörleri arasındaki benzerliği ölçerek duplikasyon kontrolü yapar
    
    Args:
        new_emb: Yeni eklenen içeriğin embedding vektörü
        existing_embs: Mevcut embedding vektörleri listesi
        threshold: Benzerlik eşik değeri (0.0-1.0)
        
    Returns:
        bool: Duplikasyon varsa True, yoksa False
    """
    if not existing_embs:
        return False
        
    logger.info(f"Async embedding tabanlı duplikasyon kontrolü: {len(existing_embs)} vektör ile karşılaştırılıyor")
    
    # Büyük koleksiyonlarda benzerlik hesaplamasını async olarak yapalım
    sims = await asyncio.gather(*[
        asyncio.to_thread(compute_cosine_similarity, new_emb, emb) 
        for emb in existing_embs
    ])
    
    max_sim = max(sims) if sims else 0
    
    logger.info(f"Async - En yüksek benzerlik skoru: {max_sim:.4f} (Eşik: {threshold:.4f})")
    return max_sim > threshold

@duplicate_error_handler
def get_duplicate_threshold_from_config(config_path='config/chunking_config.yaml'):
    """
    Konfigürasyon dosyasından duplikasyon eşik değerini okur
    
    Args:
        config_path: Konfigürasyon dosyasının yolu
        
    Returns:
        float: Duplikasyon eşik değeri (0.0-1.0)
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        threshold = config.get('chunking', {}).get('duplicate_threshold', 0.80)
        logger.info(f"Duplikasyon eşik değeri: {threshold}")
        return float(threshold)
    except Exception as e:
        logger.error(f"Konfigürasyon dosyası okunurken hata oluştu: {e}")
        return 0.80 