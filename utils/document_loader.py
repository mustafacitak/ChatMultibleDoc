import os
from typing import List, Dict, Union, Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd
from bs4 import BeautifulSoup
import requests
import ssl
import warnings

# SSL sertifika sorunlarını atlayalım
ssl._create_default_https_context = ssl._create_unverified_context

def load_document(file_path: str) -> List[Document]:
    """
    Verilen dosya yolundaki belgeyi yükler ve Document nesneleri listesi döndürür.
    Desteklenen formatlar: PDF, DOCX, CSV, XLSX
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    documents = []
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        
        elif ext == '.docx':
            try:
                # NLTK işlevlerini kullanmadan basit DOCX okuma
                import docx
                doc = docx.Document(file_path)
                text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
                metadata = {"source": file_path}
                documents = [Document(page_content=text, metadata=metadata)]
                print("DOCX python-docx ile başarıyla okundu.")
            except ImportError:
                print("python-docx bulunamadı, UnstructuredWordDocumentLoader deneniyor...")
                try:
                    loader = UnstructuredWordDocumentLoader(file_path)
                    documents = loader.load()
                except Exception as docx_error:
                    raise Exception(f"DOCX yüklenemedi: {str(docx_error)}")
        
        elif ext == '.csv':
            # Farklı encoding türlerini dene
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            success = False
            
            for encoding in encodings:
                try:
                    # Pandas ile CSV dosyasını okuyalım
                    df = pd.read_csv(file_path, encoding=encoding)
                    for i, row in df.iterrows():
                        # Her satırı "Kolon: Değer" formatında metne dönüştür
                        content = "\n".join([f"{col}: {str(val)}" for col, val in row.items() if pd.notna(val)])
                        metadata = {"source": file_path, "row": i}
                        documents.append(Document(page_content=content, metadata=metadata))
                    success = True
                    print(f"CSV başarıyla {encoding} encodingle okundu.")
                    break
                except UnicodeDecodeError:
                    print(f"CSV {encoding} ile okunamadı, diğer encoding deneniyor...")
                    continue
                except Exception as csv_error:
                    print(f"CSV {encoding} ile okuma hatası: {str(csv_error)}")
                    # Son encoding denemesiyse hata ver
                    if encoding == encodings[-1]:
                        raise Exception(f"CSV dosyası hiçbir encoding ile okunamadı")
            
            # Pandas ile okunamadıysa CSVLoader'ı dene
            if not success and not documents:
                try:
                    for encoding in encodings:
                        try:
                            loader = CSVLoader(file_path=file_path, encoding=encoding)
                            documents = loader.load()
                            print(f"CSV CSVLoader ile {encoding} encodingle başarıyla okundu.")
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            if encoding == encodings[-1]:
                                raise Exception(f"Error loading {file_path}: {str(e)}")
                except Exception as e:
                    raise Exception(f"CSV yüklenemedi: {str(e)}")
            
        elif ext == '.xlsx':
            try:
                # Excel dosyasını pandas ile okuyup her satırı bir Document'a dönüştürüyoruz
                df = pd.read_excel(file_path)
                for i, row in df.iterrows():
                    # Her satırı "Kolon: Değer" formatında metne dönüştür
                    content = "\n".join([f"{col}: {str(val)}" for col, val in row.items() if pd.notna(val)])
                    metadata = {"source": file_path, "row": i}
                    documents.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                raise Exception(f"Excel dosyası okunurken hata: {str(e)}")
        
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {ext}")
            
    except Exception as e:
        raise Exception(f"Belge yüklenirken hata oluştu: {str(e)}")
        
    return documents

def load_url_content(url: str) -> List[Document]:
    """
    Verilen URL'deki içeriği çeker ve Document nesnesi olarak döndürür.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # HTTP hatalarını kontrol et
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Metni ayıkla ve temizle
        text = soup.get_text(separator="\n")
        
        # Boş satırları temizle
        text = "\n".join([line for line in text.split("\n") if line.strip()])
        
        # Document nesnesi oluştur
        doc = Document(page_content=text, metadata={"source": url, "type": "webpage"})
        
        return [doc]
    
    except Exception as e:
        raise Exception(f"URL içeriği yüklenirken hata oluştu: {str(e)}")

def split_documents(documents: List[Document], 
                   chunk_size: int = 500, 
                   chunk_overlap: int = 50) -> List[Document]:
    """
    Belgeleri belirtilen boyutlarda parçalara ayırır.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    return text_splitter.split_documents(documents)

# Yeni fonksiyonlar, eski fonksiyonları çağırıp ilave işlemler yapan sarmalayıcılardır
def load_document_from_file(file_path: str, file_name: str = None) -> List[Document]:
    """
    Dosyadan belge yükler ve içeriği parçalara ayırır.
    
    Args:
        file_path: Yüklenecek dosyanın yolu
        file_name: İsteğe bağlı dosya adı (metadata için)
    
    Returns:
        Parçalanmış belge listesi
    """
    # Belgeyi yükle
    documents = load_document(file_path)
    
    # Metadata'ya dosya adını ekle
    if file_name:
        for doc in documents:
            doc.metadata["filename"] = file_name
    
    # Belgeleri parçala (500 karakter, 50 karakter örtüşme) - daha küçük parçalar için değiştirdik
    chunked_docs = split_documents(documents, chunk_size=500, chunk_overlap=50)
    
    return chunked_docs

def load_document_from_url(url: str) -> List[Document]:
    """
    URL'den içerik yükler ve parçalara ayırır.
    
    Args:
        url: İçerik yüklenecek URL
    
    Returns:
        Parçalanmış belge listesi
    """
    # URL içeriğini yükle
    documents = load_url_content(url)
    
    # Belgeleri parçala (500 karakter, 50 karakter örtüşme) - daha küçük parçalar için değiştirdik
    chunked_docs = split_documents(documents, chunk_size=500, chunk_overlap=50)
    
    return chunked_docs 