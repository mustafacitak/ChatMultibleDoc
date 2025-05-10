# DocChat - Belge Analiz ve Sohbet Sistemi

DocChat, belgelerinizi yapay zeka ile analiz etmenizi ve sohbet etmenizi sağlayan LangChain ve Streamlit tabanlı bir RAG (Retrieval-Augmented Generation) uygulamasıdır.

## Özellikler

- 📄 Farklı belge formatları desteği (PDF, DOCX, CSV, XLSX)
- 🔗 Web sayfası içeriği analizi
- 💬 Belgeler üzerinde yapay zeka ile sohbet
- 📊 Belge analizi ve özet çıkarma
- 🔍 Semantik arama ile içerik bulma
- ⚙️ Özelleştirilebilir sistem promptları
- 🌐 Çokdilli destek ('paraphrase-multilingual-MiniLM-L12-v2' embedding modeli ile)

## Dosya Yapısı

```
dochat/
├── app.py                  # Ana uygulama giriş noktası
├── .env                    # Ortam değişkenleri (API anahtarı vb.)
├── .env.example            # Örnek .env dosyası
├── config/
│   └── system_prompt.yaml  # Sistem promptları
├── db/                     # Vektör veritabanı depolama alanı
├── pages/
│   ├── 01_dokuman_yukle.py # Belge yükleme sayfası
│   ├── 02_chat.py          # Sohbet arayüzü sayfası
│   └── 03_ayarlar.py       # Ayarlar sayfası
├── utils/
│   ├── __init__.py         # Python modülü tanımı
│   ├── config.py           # Uygulama yapılandırma değişkenleri
│   ├── document_loader.py  # Belge yükleme ve işleme fonksiyonları
│   ├── embeddings.py       # Gömme modelleri için yardımcılar
│   ├── langchain_helpers.py # LangChain entegrasyonu yardımcıları
│   └── rag.py              # RAG yapısı için temel fonksiyonlar
└── requirements.txt        # Bağımlılıklar
```

## Kurulum

1. Projeyi indirin:
```bash
git clone https://github.com/kullanici/dochat.git
cd dochat
```

2. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

3. API anahtarınızı ayarlayın:
   - `.env.example` dosyasını `.env` olarak kopyalayın ve Google Gemini API anahtarınızı ekleyin
   ```bash
   cp .env.example .env
   # .env dosyasını düzenleyerek API anahtarınızı ekleyin
   ```
   - [Google AI Studio](https://ai.google.dev/)'dan API anahtarı edinin
   - **Not:** Embedding için artık API anahtarına ihtiyaç duyulmaz, yerel model kullanılır

## Kullanım

1. Uygulamayı başlatın:
```bash
streamlit run app.py
```

2. Tarayıcıda açılan arayüz üzerinden:
   - "Doküman Yükle" sayfasından belgelerinizi ekleyin
   - "Sohbet" sayfasında belgeleriniz hakkında sorular sorun
   - "Ayarlar" sayfasından sistem promptlarını özelleştirin

## Teknolojiler

- **LangChain**: RAG mimarisi için temel çerçeve
- **Google Gemini 2.0 Flash**: Yapay zeka dil modeli
- **SentenceTransformers**: Çokdilli yerel embeddings modeli
- **ChromaDB**: Vektör veritabanı
- **Streamlit**: Kullanıcı arayüzü

## Gereksinimler

Uygulama şu paketlere ihtiyaç duyar:
- Streamlit >= 1.42.0
- LangChain >= 0.1.0
- ChromaDB >= 0.4.18
- Google Generative AI >= 0.3.1
- SentenceTransformers >= 2.2.2
- PyYAML >= 6.0
- Python-dotenv >= 1.0.0
- ve diğer paketler `requirements.txt` dosyasında listelenmiştir

## API Kullanımı

Uygulama Google Gemini 2.0 Flash modeli ile çalışır. API anahtarı `.env` dosyasında tanımlanmalıdır.
Embedding işlemleri için yerel 'paraphrase-multilingual-MiniLM-L12-v2' modeli kullanılır ve API anahtarı gerektirmez.

## Özelleştirme

Sistem promptlarını `Ayarlar` sayfasından düzenleyebilir veya doğrudan `config/system_prompt.yaml` dosyasını değiştirebilirsiniz.

## Önemli Not

Uygulamanın önceki versiyonlarında Google API'yi kullanan vektör veritabanları oluşturulduysa, yeni yerel model farklı bir vektör uzayı kullanabileceğinden, bu veritabanları yeniden oluşturulmalıdır. Mevcut koleksiyonları silip (`db` klasöründen) belgeleri yeniden yükleyerek yeni model ile tutarlı sonuçlar elde edebilirsiniz.

## İletişim

Sorular, öneriler veya geri bildirimler için: example@example.com

---

DocChat v1.0 | RAG Tabanlı Doküman Analiz Sistemi
