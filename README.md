PDF Dosyası İçeriği ile Etkileşimli Sohbet Arayüzü
Bu uygulama, kullanıcıların PDF dosyalarını yükleyip içeriklerinde arama yapmalarını ve belgeyle etkileşime girmelerini sağlar. Ayrıca, kullanıcıların belgeyle ilgili sorular sormasına olanak tanır ve bu sorulara cevap alabilirler.

Kullanılan Teknolojiler
Python
Streamlit: Kullanıcı arayüzü oluşturmak için kullanılmıştır.
PyPDF2: PDF dosyalarını işlemek ve içeriklerini çıkarmak için kullanılmıştır.
OpenAI API: Belgeye yönelik sorulara cevap üretmek için kullanılmıştır.
FAISS: Metinler arasında benzerlik aramak için kullanılmıştır.
Nasıl Kullanılır?
"Upload your files." butonuna tıklayarak PDF dosyalarını yükleyin.
"Ask questions about your file:" alanına belgeyle ilgili bir soru yazın.
"Send" butonuna tıklayarak sorunuzu gönderin.
Cevabı ekranda görüntüleyin.
Ayrıca, yüklediğiniz belgelerin içeriğini ve önizlemesini görüntüleyebilirsiniz.

Kurulum
Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

Repoyu klonlayın:
bash
git clone https://github.com/your_username/your_project.git
Proje dizinine gidin:
bash
cd your_project
Gerekli Python paketlerini yükleyin:
bash
Copy code
pip install -r requirements.txt
Uygulamayı başlatın:
bash
Copy code
streamlit run app.py
Artık tarayıcınızdan http://localhost:8501 adresine giderek uygulamayı kullanabilirsiniz.
