import os
from typing import Optional, List, Dict
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from utils.config import MODEL_NAME, get_api_key
from langchain_core.language_models import BaseChatModel

def load_system_prompts(file_path: str = "config/system_prompt.yaml") -> dict:
    """
    System promptlarını YAML dosyasından yükler
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompts = yaml.safe_load(file)
        return prompts
    except Exception as e:
        raise Exception(f"System promptları yüklenirken hata: {str(e)}")

def get_llm(temperature: float = 0.3, system_prompt: Optional[str] = None, safety_settings: Optional[Dict] = None) -> BaseChatModel:
    """
    LangChain ile Gemini modeline erişim sağlayan bir LLM nesnesi döndürür.
    API anahtarı .env dosyasından alınır.
    
    Args:
        temperature: Modelin yaratıcılık seviyesini belirleyen sıcaklık parametresi
        system_prompt: İsteğe bağlı sistem prompt
        safety_settings: Gemini modelinin safety ayarları
        
    Returns:
        LangChain uyumlu bir ChatModel nesnesi
    
    Not:
        Bu fonksiyon şu anda Google Gemini modelini kullanıyor.
        embeddings.py'de olduğu gibi yerel bir alternatif modele geçiş yapılabilir.
    """
    try:
        # API anahtarı kontrol et - anahtarsız çalışan lokal modele geçiş yapabilirsiniz
        api_key = get_api_key()
        if not api_key:
            print("Google API anahtarı bulunamadı! get_llm() fonksiyonunda yerel bir model kullanmayı düşünebilirsiniz.")
        
        # Varsayılan safety ayarlarını tanımla (eğer kullanıcı tarafından belirtilmemişse)
        default_safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
        }
        
        # Kullanıcı tarafından sağlanan ayarlar veya varsayılan ayarları kullan
        safety_settings = safety_settings or default_safety_settings
        
        # LLM'i başlat
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            temperature=temperature,
            convert_system_message_to_human=True,  # Gemini ile daha iyi çalışabilir
            safety_settings=safety_settings
        )
        print(f"Google Gemini LLM başlatıldı: {MODEL_NAME}")
        return llm
    except Exception as e:
        # Hata durumunda alternatif model kullanılabilir
        # Örneğin: Burada yerel bir LLM veya farklı bir servisi kullanabilirsiniz
        raise Exception(f"LLM başlatılırken hata: {str(e)}\nAlternatif bir model kullanmayı düşünebilirsiniz.")

def format_sources(source_documents) -> str:
    """
    Kaynak belgeleri formatlar ve hangi belgelerden bilgi kullanıldığını gösteren bir metin döndürür
    
    Args:
        source_documents: LangChain tarafından döndürülen kaynak belgelerin listesi
        
    Returns:
        Formatlanmış kaynak metni
    """
    sources = []
    for i, doc in enumerate(source_documents):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        row = doc.metadata.get("row", "")
        
        if page:
            source_info = f"{source} (Page: {page})"
        elif row:
            source_info = f"{source} (Row: {row})"
        else:
            source_info = source
            
        if source_info not in sources:
            sources.append(source_info)
    
    if not sources:
        return ""
    
    formatted_sources = "Sources:\n" + "\n".join([f"- {s}" for s in sources])
    return formatted_sources 