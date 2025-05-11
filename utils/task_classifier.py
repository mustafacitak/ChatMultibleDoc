import os
from typing import Dict, Any, Tuple, Optional, List
from langchain_core.prompts import PromptTemplate
from utils.langchain_helpers import get_llm, load_system_prompts
from langchain_google_genai import HarmCategory, HarmBlockThreshold

# Görev türleri
TASK_TYPES = {
    "default_chat": "Genel sohbet veya belirli bir kategoriye girmeyen sorular",
    "document_analysis": "Belge içeriğinin analizi ve yapısını anlama",
    "question_answering": "Belirli sorulara cevap arama",
    "code_explanation": "Kod içeriğini açıklama veya kodla ilgili sorular",
    "summarization": "Belge içeriğini özetleme",
    "data_analysis": "Veri kümesi analizi, istatistik ve eğilim tespiti",
    "web_content_summary": "Web sayfası içeriğini özetleme ve analiz etme",
    "decision_support": "Karar verme süreçlerine destek ve alternatif analizi",
    "information_retrieval": "Belge ve verileri düzenleme, arama ve erişim",
    "error_detection": "Metin veya kod içindeki hataları tespit etme ve düzeltme"
}

def get_classifier_llm():
    """
    Sınıflandırma ve sorgu zenginleştirme için optimize edilmiş düşük temperature değerine sahip LLM
    
    Returns:
        Deterministik yanıtlar için optimize edilmiş LLM
    """
    # Safety ayarları - get_llm ile aynı formatta
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
    }
    
    # Çok düşük temperature (0.1) ile daha deterministik yanıtlar
    return get_llm(temperature=0.1, safety_settings=safety_settings)

def get_qa_llm():
    """
    Soru-cevap için optimize edilmiş orta temperature değerine sahip LLM
    
    Returns:
        Soru-cevap için optimize edilmiş LLM
    """
    # Safety ayarları - get_llm ile aynı formatta
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
    }
    
    # Orta temperature (0.4) ile dengeli çeşitlilik ve doğruluk
    return get_llm(temperature=0.4, safety_settings=safety_settings)

def classify_task(query: str, chat_history: Optional[List[Dict]] = None) -> str:
    """
    Kullanıcı sorgusunu analiz ederek hangi göreve ait olduğunu belirler.
    
    Args:
        query: Kullanıcı sorgusu
        chat_history: Önceki mesajlaşma geçmişi (opsiyonel)
    
    Returns:
        Belirlenen görev türü (default_chat, document_analysis, vb.)
    """
    # Sınıflandırma için düşük temperature kullanan LLM
    llm = get_classifier_llm()
    
    # Geçmiş sohbet bağlamını oluştur (varsa)
    context = ""
    if chat_history and len(chat_history) > 0:
        last_msgs = chat_history[-3:] if len(chat_history) > 3 else chat_history
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_msgs])
    
    # Sınıflandırma promptu
    classification_prompt = PromptTemplate(
        template="""Sen bir görev sınıflandırma asistanısın. Kullanıcının sorgusuna bakarak, 
bu sorgunun hangi tür göreve ait olduğunu belirlemen gerekiyor.

Aşağıdaki görev türlerinden BİRİNİ seç:

{task_types}

{context_section}

Kullanıcı sorgusu: {query}

Dikkat! Cevabında sadece görev türünü yaz (default_chat, document_analysis, question_answering, code_explanation, summarization, data_analysis, web_content_summary, decision_support, information_retrieval veya error_detection).
Açıklama yapma, sadece görev türünü döndür.
Görev türü:""",
        input_variables=["query", "task_types", "context_section"]
    )
    
    # Prompt değişkenlerini hazırla
    task_types_str = "\n".join([f"- {task}: {desc}" for task, desc in TASK_TYPES.items()])
    context_section = f"\nSon sohbet mesajları:\n{context}" if context else ""
    
    # LLM'e sorguyu gönder
    response = llm.invoke(
        classification_prompt.format(
            query=query,
            task_types=task_types_str,
            context_section=context_section
        )
    )
    
    # Yanıtı temizle ve doğrula
    task_type = response.content.strip().lower()
    
    # Eğer yanıt geçerli bir görev türü değilse, varsayılan olarak 'default_chat' kullan
    if task_type not in TASK_TYPES:
        print(f"Geçersiz görev türü: {task_type}, varsayılan 'default_chat' kullanılıyor")
        task_type = "default_chat"
    
    return task_type

def improve_query(query: str, task_type: str, doc_context: Optional[str] = None) -> str:
    """
    Kullanıcı sorgusunu iyileştirir ve daha spesifik hale getirir.
    
    Args:
        query: Orijinal kullanıcı sorgusu
        task_type: Belirlenen görev türü
        doc_context: Belge içeriğinden bir örnek (varsa)
    
    Returns:
        İyileştirilmiş sorgu
    """
    # Sorgu zaten spesifik ve detaylıysa, değiştirme
    if len(query.split()) > 10:
        return query
    
    # Zenginleştirme için düşük temperature kullanan LLM    
    llm = get_classifier_llm()
    
    # Belge bağlamı varsa ekle, yoksa boş bırak
    context_info = f"\nBelge içeriğinden örnek:\n{doc_context}" if doc_context else ""
    
    # Görev türüne özel iyileştirme mesajı
    task_specific_hint = ""
    if task_type == "document_analysis":
        task_specific_hint = "Belgenin yapısı, ana başlıkları ve içeriği hakkında daha spesifik sorular oluştur."
    elif task_type == "question_answering":
        task_specific_hint = "Belge içeriğindeki bilgilerle ilgili net ve kesin sorular oluştur."
    elif task_type == "code_explanation":
        task_specific_hint = "Kodun işlevi, algoritması veya belirli bölümleri hakkında detaylı sorular oluştur."
    elif task_type == "summarization":
        task_specific_hint = "Belgenin tümü veya belirli bölümleri için özetleme isteği oluştur."
    elif task_type == "data_analysis":
        task_specific_hint = "Veri kümesi hakkında istatistiksel analiz, eğilim tespiti veya grafik önerisi isteği oluştur."
    elif task_type == "web_content_summary":
        task_specific_hint = "Web sayfasının ana fikirlerini ve önemli noktalarını vurgulayan özetleme isteği oluştur."
    elif task_type == "decision_support":
        task_specific_hint = "Karşılaştırmalı analiz, senaryo değerlendirme veya karar destek isteği oluştur."
    elif task_type == "information_retrieval":
        task_specific_hint = "Belirli bilgileri bulma, kategorilere ayırma veya arama kriterlerini belirten istekler oluştur."
    elif task_type == "error_detection":
        task_specific_hint = "Metin veya kod içindeki hataları tespit etme ve düzeltme önerisi isteği oluştur."
    
    # İyileştirme promptu
    improvement_prompt = PromptTemplate(
        template="""Kullanıcı sorgusunu iyileştir ve daha spesifik hale getir. Orijinal amaç ve anlam korunmalı.
Görev türü: {task_type}
{task_specific_hint}
{context_info}

Orijinal sorgu: {query}

İyileştirilmiş sorgu:""",
        input_variables=["query", "task_type", "task_specific_hint", "context_info"]
    )
    
    # LLM'e sorguyu gönder
    response = llm.invoke(
        improvement_prompt.format(
            query=query,
            task_type=task_type,
            task_specific_hint=task_specific_hint,
            context_info=context_info
        )
    )
    
    improved_query = response.content.strip()
    
    # Eğer iyileştirme başarısız olursa veya çok uzunsa, orijinal sorguyu kullan
    if not improved_query or len(improved_query) > len(query) * 3:
        return query
    
    return improved_query

def get_task_based_rag(query: str, chat_history: Optional[List[Dict]] = None, doc_sample: Optional[str] = None) -> Tuple[str, str]:
    """
    Kullanıcı sorgusunu görev türüne göre sınıflandırır ve iyileştirir.
    
    Args:
        query: Kullanıcı sorgusu
        chat_history: Önceki chat geçmişi (opsiyonel)
        doc_sample: Belge içeriğinden örnek (opsiyonel)
    
    Returns:
        Tuple[str, str]: (görev türü, iyileştirilmiş sorgu)
    """
    # Aşama 1: Görevi sınıflandır
    task_type = classify_task(query, chat_history)
    
    # Aşama 2: Sorguyu iyileştir
    improved_query = improve_query(query, task_type, doc_sample)
    
    print(f"Orijinal sorgu: '{query}'")
    print(f"Belirlenen görev: {task_type}")
    print(f"İyileştirilmiş sorgu: '{improved_query}'")
    
    return task_type, improved_query 