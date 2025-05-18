import os
import yaml
from loguru import logger
import json
import google.generativeai as genai
from utils.embedding_utils import get_config_param

def get_temperature_from_config(purpose='response_generation', config_path='config/chunking_config.yaml', default=0.1):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        temperature = config.get('gemini', {}).get('temperature', {}).get(purpose, default)
        return float(temperature)
    except Exception as e:
        logger.error(f"Config okuma hatası: {e}")
        return default

def get_model_name_from_config(config_path='config/chunking_config.yaml', default="gemini-pro"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        model_name = config.get('gemini', {}).get('model_name', default)
        return model_name
    except Exception as e:
        logger.error(f"Config okuma hatası: {e}")
        return default

def get_gemini_model(api_key, temperature=None, purpose='response_generation', safety_settings=None, model_name=None):
    genai.configure(api_key=api_key)
    if temperature is None:
        temperature = get_temperature_from_config(purpose)
    if model_name is None:
        model_name = get_model_name_from_config()
    if safety_settings is None:
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

def compress_context(context, query, gemini_api_key):
    """
    Büyük bağlamı sorguya uygun şekilde sıkıştır
    Not: Bu fonksiyon, Context Distillation tamamlandıktan sonra ihtiyaç duyulursa çağrılır
    Context Distillation zaten metni optimize eder, ancak tüm bağlam hala çok büyükse
    bu fonksiyon ek bir sıkıştırma gerçekleştirir.
    """
    # Bağlam sıkıştırma aktif mi kontrol et
    if not get_config_param('context_optimization.compress_context', True):
        return context
        
    # Çok kısa bağlamlar için sıkıştırma yapmaya gerek yok
    max_size = get_config_param('chunking.max_context_size', 15000)
    if len(context) <= max_size:
        return context
    
    # Context Distillation yapıldığında belirtilir
    distillation_applied = "[DISTILLED]" in context
    
    # Bağlamı sıkıştırmak için prompt hazırla
    sanitized_query = query.encode('utf-8', errors='replace').decode('utf-8') if isinstance(query, str) else str(query)
    sanitized_context = context.encode('utf-8', errors='replace').decode('utf-8') if isinstance(context, str) else str(context)
    
    # Eğer context zaten distill edilmişse daha basit bir sıkıştırma yapalım
    if distillation_applied:
        compression_prompt = f"""
        Bu metin zaten bir kez damıtılmış (distilled) bağlamdır. Metin hala çok uzundur.
        Lütfen metni daha da sıkıştır ama önemli bilgileri koru.
        
        Soru: {sanitized_query}
        
        Bağlam:
        {sanitized_context[:max_size * 2]}
        
        Sıkıştırılmış metin en fazla {max_size} karakter olmalı.
        """
    else:
        # Normal sıkıştırma
        compression_prompt = f"""
        Aşağıdaki metin, başvuru kaynaklarından oluşan bir bağlamdır. Bu bağlamı belirtilen soru için önemli olan bilgilere odaklanarak sıkıştır.
        
        Soru: {sanitized_query}
        
        Bağlam:
        {sanitized_context[:max_size * 2]}
        
        Metni önem sırasına göre özetle, önemli bilgileri koruyarak sıkıştır. Özetlenen metin en fazla {max_size} karakter olmalı.
        """
    
    try:
        logger.info(f"[CONTEXT_COMPRESSION] Bağlam sıkıştırılıyor: {len(context)} -> hedef: {max_size} karakter")
        model = get_gemini_model(gemini_api_key, purpose='response_generation')
        response = model.generate_content(compression_prompt)
        compressed = response.text.strip()
        compressed = compressed.encode('utf-8', errors='replace').decode('utf-8')
        logger.info(f"[CONTEXT_COMPRESSION] Bağlam sıkıştırıldı: {len(compressed)} karakter")
        
        # Sıkıştırılmış metni işaretle
        if not distillation_applied:
            compressed = "[COMPRESSED] " + compressed
        return compressed
    except Exception as e:
        logger.error(f"[CONTEXT_COMPRESSION] Bağlam sıkıştırma hatası: {e}")
        return context[:max_size]  # Hata durumunda basitçe kes

def generate_initial_response(query, context, gemini_api_key):
    # Token optimizasyonu için bağlamı sıkıştır
    if get_config_param('context_optimization.enabled', True):
        context = compress_context(context, query, gemini_api_key)
    
    # Sanitize input to avoid encoding issues
    sanitized_query = query.encode('utf-8', errors='replace').decode('utf-8') if isinstance(query, str) else str(query)
    sanitized_context = context.encode('utf-8', errors='replace').decode('utf-8') if isinstance(context, str) else str(context)
    
    prompt = f"""
    Soru: {sanitized_query}
    Bağlam: {sanitized_context}
    Lütfen soruya bağlamdan faydalanarak ilk yanıtı üret.
    """
    try:
        logger.info(f"[RESPONSE_GENERATION] Gemini'ye istek gönderiliyor.\nPrompt uzunluğu: {len(prompt)} karakter\nModel: {get_model_name_from_config()}\nAPI_KEY: {gemini_api_key[:6]}***")
        model = get_gemini_model(gemini_api_key, purpose='response_generation')
        response = model.generate_content(prompt)
        answer = response.text.strip()
        # Ensure response is properly encoded
        answer = answer.encode('utf-8', errors='replace').decode('utf-8')
        logger.info(f"[RESPONSE_GENERATION] Gemini'den dönen yanıt uzunluğu: {len(answer)} karakter")
        return answer
    except Exception as e:
        logger.error(f"[RESPONSE_GENERATION] Yanıt üretme hatası: {e}")
        return ""

def breakdown_reasoning_steps(query, context, initial_response, gemini_api_key):
    # Token optimizasyonu: İleri aşamalarda daha az bağlam kullan
    if get_config_param('context_optimization.progressive_context', True):
        max_context_size = get_config_param('chunking.max_context_size', 15000) // 2
        if len(context) > max_context_size:
            context = context[:max_context_size]
            logger.info(f"[OPTIMIZATION] İkinci aşama için bağlam boyutu azaltıldı: {len(context)} karakter")
    
    # Token optimizasyonu: Bağlamı tekrar göndermek yerine, ilk yanıtı kullan
    if get_config_param('context_optimization.reuse_context', True):
        # Bağlam yerine ilk yanıtı referans al
        sanitized_query = query.encode('utf-8', errors='replace').decode('utf-8') if isinstance(query, str) else str(query)
        sanitized_response = initial_response.encode('utf-8', errors='replace').decode('utf-8') if isinstance(initial_response, str) else str(initial_response)
        
        prompt = f"""
        [Adım 1] Sorgunun ana temasını belirle: '{sanitized_query}'
        [Adım 2] İlk yanıttan yola çıkarak mantıksal adımları belirle: {sanitized_response}
        [Adım 3] Mantıksal çıkarım yap ve sonuç oluştur
        """
    else:
        # Eski yöntem: Tam bağlamı gönder
        sanitized_query = query.encode('utf-8', errors='replace').decode('utf-8') if isinstance(query, str) else str(query)
        sanitized_context = context.encode('utf-8', errors='replace').decode('utf-8') if isinstance(context, str) else str(context)
        
        prompt = f"""
        [Adım 1] Sorgunun ana temasını belirle: '{sanitized_query}'
        [Adım 2] Context'teki ilgili bilgileri işaretle: {sanitized_context}
        [Adım 3] Mantıksal çıkarım yap ve sonuç oluştur
        """
    
    try:
        logger.info(f"[REASONING_STEPS] Gemini'ye istek gönderiliyor.\nPrompt uzunluğu: {len(prompt)} karakter\nModel: {get_model_name_from_config()}\nAPI_KEY: {gemini_api_key[:6]}***")
        model = get_gemini_model(gemini_api_key, purpose='reasoning_steps')
        response = model.generate_content(prompt)
        steps = response.text.strip()
        # Ensure response is properly encoded
        steps = steps.encode('utf-8', errors='replace').decode('utf-8')
        logger.info(f"[REASONING_STEPS] Gemini'den dönen yanıt uzunluğu: {len(steps)} karakter")
        return steps
    except Exception as e:
        logger.error(f"[REASONING_STEPS] RCoT hatası: {e}")
        return ""

def validate_response_accuracy(response, context, initial_response, gemini_api_key):
    # Token optimizasyonu: İleri aşamalarda daha az bağlam kullan
    if get_config_param('context_optimization.progressive_context', True):
        max_context_size = get_config_param('chunking.max_context_size', 15000) // 3
        if len(context) > max_context_size:
            context = context[:max_context_size]
            logger.info(f"[OPTIMIZATION] Üçüncü aşama için bağlam boyutu azaltıldı: {len(context)} karakter")
    
    # Token optimizasyonu: Bağlamı tekrar göndermek yerine, ilk yanıtı kullan
    if get_config_param('context_optimization.reuse_context', True):
        # Bağlam yerine yanıtları referans al
        sanitized_response = response.encode('utf-8', errors='replace').decode('utf-8') if isinstance(response, str) else str(response)
        sanitized_initial = initial_response.encode('utf-8', errors='replace').decode('utf-8') if isinstance(initial_response, str) else str(initial_response)
        
        prompt = f"""
        Yanıt: {sanitized_response}
        İlk yanıt: {sanitized_initial}
        
        Yanıtın tutarlı olup olmadığını kontrol et. İlk yanıtla çelişki var mı? Eksik bilgi varsa belirt.
        """
    else:
        # Eski yöntem: Tam bağlamı gönder
        sanitized_response = response.encode('utf-8', errors='replace').decode('utf-8') if isinstance(response, str) else str(response)
        sanitized_context = context.encode('utf-8', errors='replace').decode('utf-8') if isinstance(context, str) else str(context)
        
        prompt = f"""
        Yanıt: {sanitized_response}
        Bağlam: {sanitized_context}
        Yanıtın bağlamla tutarlı olup olmadığını kontrol et. Eksik bilgi varsa belirt.
        """
    
    try:
        logger.info(f"[VALIDATION] Gemini'ye istek gönderiliyor.\nPrompt uzunluğu: {len(prompt)} karakter\nModel: {get_model_name_from_config()}\nAPI_KEY: {gemini_api_key[:6]}***")
        model = get_gemini_model(gemini_api_key, purpose='validation')
        model_response = model.generate_content(prompt)
        validation = model_response.text.strip()
        # Ensure response is properly encoded
        validation = validation.encode('utf-8', errors='replace').decode('utf-8')
        logger.info(f"[VALIDATION] Gemini'den dönen yanıt uzunluğu: {len(validation)} karakter")
        return validation
    except Exception as e:
        logger.error(f"[VALIDATION] Doğrulama hatası: {e}")
        return ""

def update_response(original_resp, missing_info):
    # Sanitize inputs
    sanitized_original = original_resp.encode('utf-8', errors='replace').decode('utf-8') if isinstance(original_resp, str) else str(original_resp)
    sanitized_missing = missing_info.encode('utf-8', errors='replace').decode('utf-8') if isinstance(missing_info, str) else str(missing_info)
    
    updated = sanitized_original + f"\n\n[Güncelleme] Ek bilgi eklendi: {sanitized_missing}"
    logger.info("Yanıt güncellendi.")
    return updated

def save_to_memory_log(memory_path, data):
    os.makedirs(os.path.dirname(memory_path), exist_ok=True)
    
    # Sanitize data to ensure proper encoding
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                data[key] = value.encode('utf-8', errors='replace').decode('utf-8')
            elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                data[key] = [x.encode('utf-8', errors='replace').decode('utf-8') for x in value]
    
    with open(memory_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    logger.info(f"Memory log'a kayıt eklendi: {memory_path}")

def run_token_optimized_rcot(query, context, gemini_api_key):
    """
    Token optimizasyonu yapılmış RCoT işlemi
    Bu fonksiyon, tüm RCoT adımlarını tek bir uyarıda birleştirerek token kullanımını optimize eder
    """
    # Bağlam optimizasyonu aktif mi?
    if not get_config_param('context_optimization.enabled', True):
        return None
        
    # Bağlamı sıkıştır 
    if get_config_param('context_optimization.compress_context', True):
        context = compress_context(context, query, gemini_api_key)
    
    # Sanitize input
    sanitized_query = query.encode('utf-8', errors='replace').decode('utf-8') if isinstance(query, str) else str(query)
    sanitized_context = context.encode('utf-8', errors='replace').decode('utf-8') if isinstance(context, str) else str(context)
    
    # Tek bir API çağrısıyla tüm RCoT adımlarını gerçekleştir
    combined_prompt = f"""
    GÖREV: Aşağıdaki soru için verilen bağlam bilgilerini kullanarak 3 adımlı RCoT (Düşünce Zinciri) işlemini gerçekleştir.
    
    SORU: {sanitized_query}
    
    BAĞLAM:
    {sanitized_context}
    
    Adım adım şu çıktıları üret:
    
    1. İLK_YANIT: [Soruya bağlamdan faydalanarak ilk yanıtı üret]
    
    2. DÜŞÜNCE_ADIMLAR: [İlk yanıtı üretirken kullandığın akıl yürütme adımlarını açıkla]
    
    3. DOĞRULAMA: [Yanıtın bağlamla tutarlı olup olmadığını kontrol et ve varsa eksik bilgileri belirt]
    
    Her bir adımı ayrı paragraflar halinde ve açık etiketlerle (İLK_YANIT:, DÜŞÜNCE_ADIMLAR:, DOĞRULAMA:) belirterek formatla.
    """
    
    try:
        logger.info(f"[OPTIMIZED_RCOT] Gemini'ye tek istek gönderiliyor.\nPrompt uzunluğu: {len(combined_prompt)} karakter")
        model = get_gemini_model(gemini_api_key, purpose='response_generation')
        response = model.generate_content(combined_prompt)
        result = response.text.strip()
        result = result.encode('utf-8', errors='replace').decode('utf-8')
        logger.info(f"[OPTIMIZED_RCOT] Gemini'den dönen yanıt uzunluğu: {len(result)} karakter")
        
        # Yanıtı parçalara ayır
        parts = {}
        current_part = None
        current_text = []
        
        for line in result.split('\n'):
            if line.startswith("İLK_YANIT:"):
                current_part = "initial_response"
                current_text = [line.replace("İLK_YANIT:", "").strip()]
            elif line.startswith("DÜŞÜNCE_ADIMLAR:"):
                if current_part:
                    parts[current_part] = '\n'.join(current_text)
                current_part = "reasoning_steps"
                current_text = [line.replace("DÜŞÜNCE_ADIMLAR:", "").strip()]
            elif line.startswith("DOĞRULAMA:"):
                if current_part:
                    parts[current_part] = '\n'.join(current_text)
                current_part = "validation"
                current_text = [line.replace("DOĞRULAMA:", "").strip()]
            else:
                current_text.append(line)
        
        if current_part:
            parts[current_part] = '\n'.join(current_text)
        
        logger.info(f"[OPTIMIZED_RCOT] Yanıt başarıyla parçalara ayrıldı. Parça sayısı: {len(parts)}")
        return parts
    except Exception as e:
        logger.error(f"[OPTIMIZED_RCOT] Optimize edilmiş RCoT hatası: {e}")
        return None 