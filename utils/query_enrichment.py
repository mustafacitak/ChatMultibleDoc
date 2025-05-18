import re
import yaml
from loguru import logger
import google.generativeai as genai

def get_temperature_from_config(purpose='query_enrichment', config_path='config/chunking_config.yaml', default=0.1):
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

def get_gemini_model(api_key, temperature=None, purpose='query_enrichment', safety_settings=None, model_name=None):
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

def preprocess_query(query: str) -> str:
    cleaned = re.sub(r'[^\w\s]', '', query.lower())
    logger.info(f"Sorgu ön işlemden geçti: {cleaned}")
    return cleaned

def improve_query(query: str, gemini_api_key: str) -> str:
    # Sanitize input query
    sanitized_query = query.encode('utf-8', errors='replace').decode('utf-8') if isinstance(query, str) else str(query)
    
    if len(sanitized_query.strip().split()) < 4 or len(sanitized_query) < 15:
        prompt = f"""
        Kullanıcıdan gelen sorgu: '{sanitized_query}'\n
        1. Sorgudaki yazım hatalarını düzelt.\n2. Sorgu çok kısa veya belirsizse, daha açık ve anlamlı hale getir.\n3. Sorgunun anlamını değiştirme, sadece daha iyi ifade et.\n
        Sadece iyileştirilmiş sorguyu döndür.
        """
        try:
            logger.info(f"[QUERY_ENRICHMENT] Gemini'ye istek gönderiliyor.\nPrompt uzunluğu: {len(prompt)} karakter\nModel: {get_model_name_from_config()}\nAPI_KEY: {gemini_api_key[:6]}***")
            model = get_gemini_model(gemini_api_key, purpose='query_enrichment')
            response = model.generate_content(prompt)
            improved = response.text.strip()
            # Ensure result is properly encoded
            improved = improved.encode('utf-8', errors='replace').decode('utf-8')
            logger.info(f"[QUERY_ENRICHMENT] Gemini'den dönen yanıt uzunluğu: {len(improved)} karakter")
            return improved
        except Exception as e:
            logger.error(f"[QUERY_ENRICHMENT] Sorgu iyileştirme hatası: {e}")
            return sanitized_query
    else:
        return sanitized_query 
