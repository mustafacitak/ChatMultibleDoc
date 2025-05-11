import os
import tempfile
from PIL import Image
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document

class OCRProcessor:
    """Görsellerden metin çıkarmak için OCR işlemlerini yöneten sınıf"""
    
    def __init__(self):
        """OCR işlemcisini başlat"""
        self.ocr_engine = RapidOCR()
    
    def process_image(self, image_path, source_name=None):
        """Görsellerden metin çıkar ve LangChain Document nesneleri oluştur
        
        Args:
            image_path: Görsel dosyasının yolu
            source_name: Görsel dosyasının adı (None ise image_path kullanılır)
            
        Returns:
            LangChain Document nesnelerinin listesi
        """
        # Görsel adını source_name olarak ayarla (belirtilmemişse)
        if source_name is None:
            source_name = os.path.basename(image_path)
        
        # Görseli aç
        try:
            image = Image.open(image_path)
            
            # Görselin boyutunu kontrol et, çok büyükse yeniden boyutlandır
            max_dimension = 2000  # maksimum kenar uzunluğu
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size)
            
            # Görsel dosyasını geçici olarak kaydet
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                image.save(temp_path)
            
            # OCR işlemi
            result, _ = self.ocr_engine(temp_path)
            
            # Sonuç kontrolü
            if not result:
                return []
            
            # OCR sonuçlarını birleştir
            full_text = "\n".join([line[1] for line in result])
            
            # Geçici dosyayı sil
            os.unlink(temp_path)
            
            # LangChain Document nesnesi oluştur
            return [Document(
                page_content=full_text,
                metadata={
                    "source": source_name,
                    "type": "image",
                    "file_path": image_path
                }
            )]
            
        except Exception as e:
            print(f"Image OCR error: {str(e)}")
            return []
    
    def process_pdf_images(self, pdf_path, source_name=None):
        """PDF'teki görselleri çıkar ve OCR işlemini uygula
        
        Bu fonksiyon gelecekte genişletilebilir - şu anda PDF görsellerini
        çıkarma işlemi doğrudan document_loader.py içinde yapılacak
        """
        pass  # Bu fonksiyon, PDF işleme yeteneğine göre daha sonra geliştirilecek

# Tekil bir örnek oluştur
ocr_processor = OCRProcessor() 