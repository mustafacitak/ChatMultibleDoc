from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger
import yaml
import google.generativeai as genai

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Fonksiyonun başına global bir flag ekle
_config_read_error = False

def generate_embedding(text: str):
    """
    Verilen metin için embedding vektörü oluşturur.
    Cached RAG aktifse önce önbellekten kontrol eder.
    """
    # Cached RAG entegrasyonu
    try:
        from utils.cache_utils import get_cached_embedding, cache_embedding
        
        # Önbellekten kontrol et
        cached_emb = get_cached_embedding(text)
        if cached_emb is not None:
            logger.info("Embedding önbellekten alındı.")
            return cached_emb
    except ImportError:
        # cache_utils modülü yoksa, normal işleme devam et
        pass
    
    # Binary Quantization için embedding önbelleği
    try:
        if get_config_param('binary_quantization.enabled', True, _recursion_level=1):
            from utils.binary_quantization import get_binary_quantizer
            quantizer = get_binary_quantizer()
            # Quantizer'ın cache'ine bakalım
            vector_key = str(hash(text))
            if vector_key in quantizer._vector_cache:
                cached_result = quantizer._vector_cache[vector_key]
                if cached_result.get('original') is not None:
                    logger.info("Embedding binary quantizer önbelleğinden alındı.")
                    return cached_result['original']
    except ImportError:
        pass
    
    # Önbellekte yoksa embedding oluştur
    emb = model.encode([text])[0]
    logger.info("Embedding üretildi.")
    
    # Önbelleğe kaydet
    try:
        from utils.cache_utils import cache_embedding
        cache_embedding(text, emb.tolist())
    except ImportError:
        pass
    
    return emb.tolist()

def compute_cosine_similarity(vec1, vec2, allow_binary=True):
    # Binary Quantization entegrasyonu
    if allow_binary:
        try:
            use_binary = get_config_param('binary_quantization.enabled', True)
            if use_binary:
                from utils.binary_quantization import get_binary_quantizer
                quantizer = get_binary_quantizer()
                return quantizer.compute_similarity(vec1, vec2)
        except (ImportError, Exception) as e:
            logger.debug(f"Binary quantization kullanılamadı: {e}, standart kosinüs benzerliği kullanılıyor.")
    # Standart kosinüs benzerliği
    sim = cosine_similarity([vec1], [vec2])[0][0]
    return sim

def get_config_param(param_path, default_value, config_path='config/chunking_config.yaml', _recursion_level=0):
    """
    Config dosyasından parametre değerini al. Nokta notasyonu ile iç içe parametrelere erişebilir.
    Örnek: get_config_param('chunking.top_k', 5)
    """
    global _config_read_error
    # Eğer daha önce config okuma hatası olduysa, tekrar deneme
    if _config_read_error:
        return default_value
    # Zincirleme çağrıda config dosyasını tekrar açma
    if _recursion_level > 0:
        return default_value
    # Özyineleme derinliğini sınırla
    if _recursion_level > 10:  # Maksimum özyineleme derinliği
        logger.error(f"Maksimum özyineleme derinliğine ulaşıldı: {param_path}")
        return default_value
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # Nokta notasyonu ile iç içe alanlara erişim
        keys = param_path.split('.')
        value = config
        for i, key in enumerate(keys):
            if not isinstance(value, dict):
                return default_value
            if key in value:
                value = value[key]
            else:
                return default_value
        if value is None:
            return default_value
        return value
    except Exception as e:
        logger.error(f"Config parametresi okuma hatası: {e}")
        _config_read_error = True
        return default_value

def get_top_k_from_config(config_path='config/chunking_config.yaml'):
    return get_config_param('chunking.top_k', 5, config_path=config_path, _recursion_level=1)

def get_top_k_chunks(query_emb, chunks, k=None):
    """
    Sorgu vektörüne en benzer k chunk'ı bul ve min_similarity_threshold'dan düşük olanları filtrele.
    Adaptive-RAG ve FUNNELRAG entegrasyonu içerir.
    """
    # Adaptive-RAG entegrasyonu
    try:
        if get_config_param('adaptive_rag.enabled', True, _recursion_level=1):
            from utils.adaptive_rag import get_adaptive_chunk_count
            # Sorgu metnini bulamayız, k parametresi ile devam ederiz
            if k is None:
                k = get_top_k_from_config()
                logger.info(f"Adaptive-RAG: k parametresi belirtilmediği için varsayılan değer kullanılıyor: {k}")
    except (ImportError, Exception) as e:
        logger.debug(f"Adaptive-RAG kullanılamadı: {e}")
        if k is None:
            k = get_top_k_from_config()

    # FUNNELRAG entegrasyonu
    try:
        use_funnel_rag = get_config_param('funnel_rag.enabled', True, _recursion_level=1)
        if use_funnel_rag and len(chunks) > k * 3:  # FUNNELRAG için yeterli chunk var mı?
            from utils.funnel_rag import FunnelRAG
            funnel_rag = FunnelRAG()
            result = funnel_rag.get_top_k_chunks(None, chunks, k)  # query metni burada yok, None gönderiyoruz
            
            # Minimum benzerlik eşiği kontrolü
            min_similarity = get_config_param('chunking.min_similarity_threshold', 0.3, _recursion_level=1)
            result = [chunk for chunk in result if chunk.get('similarity', 0) >= min_similarity]
            
            logger.info(f"FUNNELRAG: {len(result)} chunk filtrelendi (min benzerlik: {min_similarity})")
            return result
    except (ImportError, Exception) as e:
        logger.debug(f"FUNNELRAG kullanılamadı: {e}")
    
    # Binary Quantization entegrasyonu - benzerlik hesaplama için
    use_binary = get_config_param('binary_quantization.enabled', True, _recursion_level=1)
    try:
        if use_binary:
            from utils.binary_quantization import get_binary_quantizer
            quantizer = get_binary_quantizer()
            
            # Binary quantize query embedding
            query_quantized = quantizer.quantize(query_emb)
            
            # Binary quantize all chunk embeddings
            for chunk in chunks:
                if 'embedding' in chunk and not chunk.get('is_quantized', False):
                    chunk['quantized_embedding'] = quantizer.quantize(chunk['embedding'])
            
            # Calculate similarities using binary quantized vectors
            sims = []
            for chunk in chunks:
                if 'quantized_embedding' in chunk:
                    sim = quantizer.compute_similarity(query_quantized, chunk['quantized_embedding'])
                elif 'embedding' in chunk:
                    sim = quantizer.compute_similarity(query_quantized, chunk['embedding'])
                else:
                    sim = 0
                sims.append(sim)
    except (ImportError, Exception) as e:
        logger.debug(f"Binary Quantization kullanılamadı: {e}, standart benzerlik hesaplaması kullanılıyor.")
        # Standart benzerlik hesaplama
        sims = [compute_cosine_similarity(query_emb, c['embedding']) for c in chunks]
    
    # Minimum benzerlik eşik değerini config'ten oku
    min_similarity = get_config_param('chunking.min_similarity_threshold', 0.3, _recursion_level=1)
    
    # Skorlarla birlikte indeksleri tut
    indexed_sims = [(i, sim) for i, sim in enumerate(sims)]
    
    # Benzerlik skoruna göre sırala (büyükten küçüğe)
    indexed_sims.sort(key=lambda x: x[1], reverse=True)
    
    # İlk k öğeyi al ve minimum benzerlik eşiğini uygula
    filtered_indices = [i for i, sim in indexed_sims[:k] if sim >= min_similarity]
    
    # Seçilen chunk'ları döndür ve benzerlik skorlarını da ekle
    result = []
    for i in filtered_indices:
        chunk = chunks[i].copy()
        chunk['similarity'] = sims[i]
        result.append(chunk)
    
    logger.info(f"Top-{k} chunks filtrelendi: {len(result)} chunk seçildi (min benzerlik: {min_similarity})")
    return result

def get_gemini_api_key():
    """
    Gemini API anahtarını çevre değişkenlerinden çeker
    """
    import os
    return os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

def get_gemini_model(api_key=None, temperature=None, purpose='distillation'):
    """
    Gemini modeline bağlanır
    """
    if api_key is None:
        api_key = get_gemini_api_key()
        
    genai.configure(api_key=api_key)
    
    if temperature is None:
        temperature = get_config_param(f'gemini.temperature.{purpose}', 0.2, _recursion_level=1)
        
    model_name = get_config_param('gemini.model_name', "gemini-2.0-flash-lite", _recursion_level=1)
    
    safety_settings = {
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE"
    }
    
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config={"temperature": temperature},
        safety_settings=safety_settings
    )

def distill_chunk(chunk, query, api_key=None):
    """
    Bir chunk'ı sorguya göre damıtır/optimize eder
    """
    # Distillation aktif mi kontrol et
    if not get_config_param('context_optimization.distillation.enabled', True, _recursion_level=1):
        return chunk['text']
    
    # Preserve ratio'yu config'ten oku
    preserve_ratio = get_config_param('context_optimization.distillation.preserve_ratio', 0.7, _recursion_level=1)
    approximate_target_length = int(len(chunk['text']) * preserve_ratio)
    
    # API çağrısı için chunk ve sorguyu sanitize et
    sanitized_text = chunk['text'].encode('utf-8', errors='replace').decode('utf-8') if isinstance(chunk['text'], str) else str(chunk['text'])
    sanitized_query = query.encode('utf-8', errors='replace').decode('utf-8') if isinstance(query, str) else str(query)
    
    # Damıtma için prompt hazırla
    distillation_prompt = f"""
    Orijinal metin parçası aşağıdadır:
    ---
    {sanitized_text}
    ---
    
    Sorgu: {sanitized_query}
    
    Bu metni, sorguyla ilişkili en önemli bilgileri koruyarak daha özlü bir versiyonuna dönüştür. 
    Gereksiz detayları çıkar, anahtar bilgileri koru.
    Metin yaklaşık {approximate_target_length} karakter uzunluğunda olmalı.
    Yanıtında kesinlikle sadece damıtılmış metni döndür, başka açıklama ekleme.
    """
    
    try:
        # Gemini modeline bağlan ve damıtma yap
        model = get_gemini_model(api_key, purpose='distillation')
        response = model.generate_content(distillation_prompt)
        distilled_text = response.text.strip()
        
        # Ensure response is properly encoded
        distilled_text = distilled_text.encode('utf-8', errors='replace').decode('utf-8')
        
        # Benzerlik kontrolü aktif mi?
        if get_config_param('context_optimization.distillation.check_similarity', True, _recursion_level=1):
            original_emb = generate_embedding(sanitized_text)
            distilled_emb = generate_embedding(distilled_text)
            similarity = compute_cosine_similarity(original_emb, distilled_emb)
            
            # Eğer benzerlik çok düşükse damıtmayı kullanma
            min_similarity = get_config_param('chunking.duplicate_threshold', 0.80, _recursion_level=1) * 0.9  # %90'ı kadar benzerlik olmalı
            if similarity < min_similarity:
                logger.warning(f"Damıtılmış metin orijinaliyle yeterince benzer değil ({similarity:.4f}). Orijinal metin kullanılıyor.")
                return sanitized_text
                
            logger.info(f"Damıtma sonrası benzerlik: {similarity:.4f}, Boyut: {len(sanitized_text)} → {len(distilled_text)} karakter")
        else:
            logger.info(f"Damıtma tamamlandı. Boyut: {len(sanitized_text)} → {len(distilled_text)} karakter")
        
        # İşlemi tanımlamak için etiket ekle
        return "[DISTILLED] " + distilled_text
    except Exception as e:
        logger.error(f"Damıtma hatası: {e}")
        return sanitized_text

def distill_chunks(chunks, query, api_key=None):
    """
    Tüm chunk'ları sorguya göre damıtır
    """
    # Distillation aktif mi kontrol et
    if not get_config_param('context_optimization.distillation.enabled', True, _recursion_level=1):
        return chunks
    
    # Her chunk'ı damıt
    distilled_chunks = []
    for chunk in chunks:
        # Chunk'ın kopyasını al
        distilled_chunk = chunk.copy()
        
        # Damıtılmış metni üret ve chunk'a ekle
        distilled_text = distill_chunk(chunk, query, api_key)
        
        # Damıtılmış metni kaydet
        distilled_chunk['text'] = distilled_text
        distilled_chunk['is_distilled'] = True
        
        distilled_chunks.append(distilled_chunk)
        
    logger.info(f"{len(distilled_chunks)} chunk damıtıldı.")
    return distilled_chunks

def prepare_context_for_llm(chunks, query=None, max_size=None, api_key=None):
    """
    Bağlam boyutunu sınırla ve benzerlik skoruna göre sırala
    Context Distillation uygulanması
    Adaptive-RAG için dinamik bağlam boyutu ayarlaması
    """
    # Adaptive-RAG ile dinamik bağlam boyutu
    try:
        if query and get_config_param('adaptive_rag.enabled', True, _recursion_level=1) and get_config_param('adaptive_rag.adaptive_context_sizing', True, _recursion_level=1):
            from utils.adaptive_rag import get_adaptive_context_size
            max_size = get_adaptive_context_size(query, max_size or get_config_param('chunking.max_context_size', 15000, _recursion_level=1))
            logger.info(f"Adaptive-RAG: Dinamik olarak ayarlanan bağlam boyutu: {max_size} karakter")
    except (ImportError, Exception) as e:
        logger.debug(f"Adaptive-RAG bağlam boyutu ayarlaması yapılamadı: {e}")
        if max_size is None:
            max_size = get_config_param('chunking.max_context_size', 15000, _recursion_level=1)
    
    # Cached RAG entegrasyonu - önce context önbellekten kontrol et
    if query:
        try:
            from utils.cache_utils import get_cached_context, cache_context
            
            # Query embedding'ini üretmeye çalışma - bu embedding_utils içinde çağrılıyor
            # ve muhtemelen zaten generate_embedding'de önbelleğe alınmıştır
            query_emb = None
            
            # Önbellekten context kontrolü
            cached_context = get_cached_context(query, query_emb)
            if cached_context is not None:
                logger.info("Context önbellekten alındı.")
                return cached_context
        except ImportError:
            # cache_utils modülü yoksa, normal işleme devam et
            pass
    
    # Sorgu değeri varsa distillation uygula
    if query and get_config_param('context_optimization.distillation.enabled', True, _recursion_level=1):
        # Damıtma işlemini uygula
        chunks = distill_chunks(chunks, query, api_key)
    
    # Benzerlik skoruna göre sırala
    sorted_chunks = sorted(chunks, key=lambda x: x.get('similarity', 0), reverse=True)
    
    context = ""
    for chunk in sorted_chunks:
        # Eğer bu chunk'ı eklemek maksimum boyutu aşıyorsa, durduralım
        if len(context) + len(chunk.get('text', '')) > max_size:
            break
        
        # Chunk'ı bağlama ekle
        chunk_text = chunk.get('text', '')
        if chunk_text:
            context += chunk_text + "\n\n"
    
    logger.info(f"Optimizasyon sonrası bağlam boyutu: {len(context)} karakter")
    
    # Önbelleğe kaydet
    if query:
        try:
            from utils.cache_utils import cache_context
            cache_context(query, context)
        except ImportError:
            pass
    
    return context 