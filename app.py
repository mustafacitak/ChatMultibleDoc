import streamlit as st
import os
from utils.config import init_app, DEFAULT_COLLECTION
from utils.vector_store import normalize_collection_name

# Initialize the app and perform necessary checks
app_ready = init_app()

# App title and configuration
st.set_page_config(
    page_title="DocChat - Document Analysis Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation
def main_page():
    # Main page content
    st.title("📄 DocChat")
    st.subheader("Chat with Your Documents")

    st.markdown("""
    ### Welcome to DocChat!

    With this application you can:
    - 📎 Upload PDF, DOCX, CSV, XLSX documents
    - 🔗 Add content from web pages
    - 💬 Chat with AI about your documents
    - 📝 Get document analysis and summaries

    To get started, go to the "Upload Document" page from the navigation menu.
    """)

    # API key check
    if not app_ready:
        st.error("""
        **API key not found!** 
        
        Please create a `.env` file in the root directory and add your API key like this:
        ```
        GOOGLE_API_KEY=your_api_key_here
        ```
        """)

def upload_document():
    import sys
    import tempfile
    import uuid
    from urllib.parse import urlparse
    import time

    # Add parent directory to path to import utils
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.document_loader import load_document_from_file, load_document_from_url
    from utils.embeddings import get_local_embeddings, get_cached_embeddings
    from utils.vector_store import create_or_update_collection
    from utils.config import DEFAULT_COLLECTION

    # Page title
    st.title("📄 Upload Document")

    # Initialize session state
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = DEFAULT_COLLECTION
    
    # Collection selection or creation
    with st.sidebar:
        st.subheader("Collection Settings")
        
        # Collection name input
        current_collection = st.text_input(
            "Collection Name",
            value=st.session_state.collection_name
        )
        
        # Koleksiyon adını normalize et
        normalized_collection = normalize_collection_name(current_collection)
        
        # Update collection in session state if changed
        if normalized_collection != st.session_state.collection_name:
            st.session_state.collection_name = normalized_collection
            if normalized_collection != current_collection:
                st.info(f"Koleksiyon adı normalize edildi: '{current_collection}' -> '{normalized_collection}'")
            else:
                st.info(f"Active collection changed to: {normalized_collection}")
    
    # Create tabs for different upload methods
    tab1, tab2 = st.tabs(["📎 Upload Files", "🔗 Add URL"])
    
    # File upload tab
    with tab1:
        st.subheader("Upload Files")
        
        # Supported file types explanation
        st.markdown("""
        You can upload documents of the following types:
        - 📝 **PDF** - PDF documents
        - 📝 **DOCX** - Word documents
        - 📊 **CSV** - Comma-separated values
        - 📊 **XLSX** - Excel spreadsheets
        """)
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose documents to upload",
            type=["pdf", "docx", "txt", "csv", "xlsx"],
            accept_multiple_files=True
        )
        
        # Process uploaded files
        if uploaded_files:
            if st.button("Process Files", type="primary"):
                process_uploaded_files(uploaded_files)
    
    # URL content tab
    with tab2:
        st.subheader("Add Content from URL")
        
        # URL input field
        url = st.text_input("Enter URL", placeholder="https://example.com")
        
        # Process URL content
        if url:
            if st.button("Fetch and Process URL", type="primary"):
                process_url_content(url)

def chat():
    import sys
    import time

    # Utils modüllerini içe aktar
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.embeddings import get_local_embeddings, get_cached_embeddings
    from utils.rag import load_collection, get_qa_chain
    from utils.langchain_helpers import get_llm, load_system_prompts, format_sources
    from utils.config import DEFAULT_COLLECTION, MODEL_NAME
    from utils.vector_store import normalize_collection_name

    # Başlık
    st.title("💬 Document Chat")

    # Yanıt akışını sağlayan fonksiyon
    def response_generator(result):
        words = result.split()
        for word in words:
            yield word + " "
            time.sleep(0.01)  # Akış hızı

    # Geribildirim fonksiyonu
    def submit_feedback(message_idx, feedback_type):
        if message_idx not in st.session_state.feedback:
            st.session_state.feedback[message_idx] = feedback_type
            st.toast(f"Thank you for your {feedback_type} feedback!")
        else:
            previous = st.session_state.feedback[message_idx]
            if previous != feedback_type:
                st.session_state.feedback[message_idx] = feedback_type
                st.toast(f"Feedback updated to {feedback_type}!")
            else:
                # Aynı butona ikinci kez basılırsa geribildirimi kaldır
                del st.session_state.feedback[message_idx]
                st.toast("Feedback removed!")
        
        # Sayfayı yenilemek için
        st.rerun()

    # Session state'i başlat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    if "collection_name" not in st.session_state:
        st.session_state.collection_name = DEFAULT_COLLECTION
    else:
        # Koleksiyon adını normalize et ve güncelle
        st.session_state.collection_name = normalize_collection_name(st.session_state.collection_name)

    # Koleksiyon bilgisi
    st.sidebar.info(f"Active Collection: **{st.session_state.collection_name}**")
    st.sidebar.caption(f"Using model: **{MODEL_NAME}**")
    
    # Embedding model değişikliği uyarısı
    if os.path.exists(os.path.join("db", st.session_state.collection_name)):
        st.sidebar.warning(
            "⚠️ **Uyarı**: Embedding modeli değiştirildi. Eski koleksiyonlar yeni "
            "modelle uyumlu olmayabilir. Eğer arama sonuçları beklendiği gibi "
            "değilse, koleksiyonu silip belgeleri yeniden yüklemeniz önerilir."
        )

    # Sohbet geçmişini temizleme butonu
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.rerun()

    # Geçmiş mesajları görüntüle
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Asistan mesajlarında kaynakları ve geribildirimi göster
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    st.write(message["sources"])
                
                # Geribildirim butonları
                col1, col2, col3 = st.columns([1, 1, 6])
                
                with col1:
                    current_feedback = st.session_state.feedback.get(idx, None)
                    if st.button("👍", key=f"thumbs_up_{idx}", 
                                type="primary" if current_feedback == "positive" else "secondary"):
                        submit_feedback(idx, "positive")
                
                with col2:
                    if st.button("👎", key=f"thumbs_down_{idx}", 
                                type="primary" if current_feedback == "negative" else "secondary"):
                        submit_feedback(idx, "negative")

    # Kullanıcı girdisi
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Kullanıcı mesajını göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Asistan yanıtını oluştur
        with st.chat_message("assistant"):
            try:
                # Koleksiyonu yükle
                try:
                    # Embedding modeli
                    embed_model = get_local_embeddings()
                    cached_embeddings = get_cached_embeddings(embed_model)
                    
                    # Koleksiyonu yükle
                    vectordb = load_collection(st.session_state.collection_name, cached_embeddings)
                    
                    # Sorgu için retriever
                    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
                    
                    # LLM modeli
                    llm = get_llm()
                    
                    # Sistem promptu
                    system_prompts = load_system_prompts()
                    system_prompt = system_prompts.get("default_chat", 
                        "Answer questions based on the information from the user's documents.")
                    
                    # QA zinciri
                    qa_chain = get_qa_chain(llm, retriever, system_prompt)
                    
                    # Yanıt al
                    with st.spinner("Generating response..."):
                        result = qa_chain({"query": prompt})
                        answer = result["result"]
                        source_docs = result["source_documents"]
                        
                        # Kaynakları formatla
                        sources_text = format_sources(source_docs)
                        
                        # Yanıtı akış olarak göster
                        response = st.write_stream(response_generator(answer))
                        
                        # Asistan mesajını kaydet
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources_text
                        })
                        
                        # Kaynakları göster
                        if sources_text:
                            with st.expander("Sources"):
                                st.write(sources_text)
                    
                except FileNotFoundError:
                    error_msg = "Collection not found. Please upload documents first."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "sources": ""
                    })
            
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "sources": ""
                })

def settings():
    import sys
    import yaml

    # Utils modüllerini içe aktar
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.langchain_helpers import load_system_prompts

    # Başlık
    st.title("⚙️ Settings")

    # System prompt dosyasının yolu
    system_prompt_path = "config/system_prompt.yaml"
    
    # Sekmeleri oluştur
    tab1, tab2 = st.tabs(["📝 System Prompts", "ℹ️ About"])
    
    with tab1:
        edit_system_prompts(system_prompt_path)
    
    with tab2:
        show_about()

# Initialize session state
if "collection_name" not in st.session_state:
    st.session_state.collection_name = DEFAULT_COLLECTION
    
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = "default_chat"
    
if "history" not in st.session_state:
    st.session_state.history = []

# Helper functions from pages
def process_uploaded_files(files):
    """Process multiple uploaded files"""
    import tempfile
    from utils.document_loader import load_document_from_file
    from utils.embeddings import get_local_embeddings, get_cached_embeddings
    from utils.vector_store import create_or_update_collection
    
    if not files:
        st.warning("No files uploaded.")
        return
    
    # Create embeddings
    try:
        embeddings = get_local_embeddings()
        cached_embeddings = get_cached_embeddings(embeddings)
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return
    
    # Process each file
    progress_text = "Processing files..."
    progress_bar = st.progress(0)
    
    for idx, file in enumerate(files):
        # Show current processing file
        st.info(f"Processing {file.name}...")
        
        # Create a temp file to process
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as temp_file:
            temp_file.write(file.getvalue())
            temp_path = temp_file.name
            
        try:
            # Get document chunks
            docs = load_document_from_file(temp_path, file.name)
            
            if docs:
                # Create or update vector store
                create_or_update_collection(
                    docs, 
                    cached_embeddings, 
                    st.session_state.collection_name,
                    source_name=file.name
                )
                st.success(f"Added {len(docs)} chunks from {file.name}")
            else:
                st.error(f"Failed to extract content from {file.name}")
                
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            
        finally:
            # Clean up the temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
        # Update progress bar
        progress_bar.progress((idx + 1) / len(files))
        
    progress_bar.empty()
    st.success(f"Successfully processed {len(files)} files into collection: {st.session_state.collection_name}")


def process_url_content(url):
    """Process content from a URL"""
    from urllib.parse import urlparse
    from utils.document_loader import load_document_from_url
    from utils.embeddings import get_local_embeddings, get_cached_embeddings
    from utils.vector_store import create_or_update_collection
    
    # Basic URL validation
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            st.error("Invalid URL. Please enter a complete URL including http:// or https://")
            return
    except:
        st.error("Invalid URL format")
        return
    
    # Create embeddings
    try:
        embeddings = get_local_embeddings()
        cached_embeddings = get_cached_embeddings(embeddings)
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return
    
    # Fetch and process URL content
    with st.spinner(f"Fetching content from {url}..."):
        try:
            # Get document chunks from URL
            docs = load_document_from_url(url)
            
            if docs:
                # Create or update vector store
                create_or_update_collection(
                    docs, 
                    cached_embeddings, 
                    st.session_state.collection_name,
                    source_name=url
                )
                st.success(f"Added {len(docs)} chunks from {url}")
            else:
                st.warning("No content could be extracted from the URL")
                
        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")

def edit_system_prompts(file_path):
    """System prompts editing interface"""
    import yaml
    from utils.langchain_helpers import load_system_prompts
    
    st.subheader("System Prompts")
    st.markdown("""
    System prompts help you determine how the AI model will respond.
    You can define different prompts for different tasks and use cases.
    """)
    
    # Sistem promptlarını yükle
    try:
        prompts = load_system_prompts(file_path)
        
        # Prompt listesi
        st.write("**Available Prompts:**")
        
        # Her bir prompt için düzenleme alanı
        updated_prompts = {}
        
        for prompt_name, prompt_text in prompts.items():
            with st.expander(f"{prompt_name}"):
                # Açıklama
                if prompt_name == "default_chat":
                    st.caption("This prompt is used for general chat.")
                elif prompt_name == "document_analysis":
                    st.caption("This prompt is used for document analysis.")
                elif prompt_name == "question_answering":
                    st.caption("This prompt is used for question answering.")
                elif prompt_name == "code_explanation":
                    st.caption("This prompt is used for code explanation.")
                elif prompt_name == "summarization":
                    st.caption("This prompt is used for summarization.")
                
                # Düzenleme alanı
                updated_text = st.text_area(
                    f"{prompt_name} Prompt", 
                    prompt_text, 
                    height=200,
                    key=f"prompt_{prompt_name}"
                )
                
                updated_prompts[prompt_name] = updated_text
        
        # Yeni prompt ekleme
        st.markdown("---")
        st.subheader("Add New Prompt")
        
        new_prompt_name = st.text_input("New Prompt Name", key="new_prompt_name")
        new_prompt_text = st.text_area("New Prompt Content", height=150, key="new_prompt_text")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Add New Prompt", key="add_prompt_btn"):
                if new_prompt_name and new_prompt_text:
                    # Yeni promptu ekle
                    updated_prompts[new_prompt_name] = new_prompt_text
                    
                    # YAML dosyasına kaydet
                    with open(file_path, 'w', encoding='utf-8') as file:
                        yaml.dump(updated_prompts, file, default_flow_style=False, allow_unicode=True)
                    
                    st.success(f"Prompt '{new_prompt_name}' added!")
                    st.rerun()
                else:
                    st.error("Prompt name and content cannot be empty!")
        
        # Değişiklikleri kaydet
        if st.button("Save Changes", type="primary"):
            # YAML dosyasına kaydet
            with open(file_path, 'w', encoding='utf-8') as file:
                yaml.dump(updated_prompts, file, default_flow_style=False, allow_unicode=True)
            
            st.success("Prompts saved successfully!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
        # Varsayılan promptları oluşturma seçeneği
        if st.button("Create Default Prompts"):
            create_default_prompts(file_path)
            st.success("Default prompts created!")
            st.rerun()

def create_default_prompts(file_path):
    """Creates default prompts"""
    import yaml
    
    default_prompts = {
        "default_chat": "You are a professional document analyst and assistant. "
                      "Answer questions based on the information from the user's documents. "
                      "If the information is not in the documents, clearly state this and don't make guesses. "
                      "Always provide clear, concise, and accurate answers. "
                      "Highlight important information and quote from the document content when relevant.",
        
        "document_analysis": "Analyze this document and summarize its important content. "
                           "Highlight key points, main headers, and the most critical information in the document. "
                           "Summarize in a way appropriate to the document type and identify the general purpose of the document.",
        
        "question_answering": "Answer based only on the information provided in the documents. "
                           "If you can't find the answer in the document, you should say \"This information is not available in the provided documents\". "
                           "Never make up answers from your own knowledge, only use content from the documents. "
                           "Support your answers with quotes from the document.",
        
        "code_explanation": "Technically explain the code found in the document. "
                          "Provide information about the purpose of the code, algorithms used, and potential improvements. "
                          "Explain what the functions and variables in the code example do.",
        
        "summarization": "Summarize the document concisely. Emphasize the main idea, key propositions, and conclusions. "
                       "The summary should be brief, clear, and accurately reflect the purpose and content of the document. "
                       "List the most important sections and topics."
    }
    
    # YAML dosyasına kaydet
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump(default_prompts, file, default_flow_style=False, allow_unicode=True)

def show_about():
    """Shows information about the application"""
    
    st.subheader("About DocChat")
    
    st.markdown("""
    ## DocChat v1.0

    DocChat is a RAG (Retrieval-Augmented Generation) application based on LangChain and Google Gemini.
    
    ### Features:
    
    - 📄 Support for PDF, DOCX, CSV, XLSX documents
    - 🔗 Web page content retrieval
    - 🔍 Semantic search in document contents
    - 💬 Chat with AI about your documents
    - 📊 Document analysis and summarization
    
    ### Technologies:
    
    - **LangChain**: RAG architecture setup
    - **Google Gemini API**: AI model
    - **ChromaDB**: Vector database
    - **Streamlit**: User interface
    
    ### How to Use:
    
    1. Add your documents from the "Upload Document" page
    2. Ask questions about your documents on the "Chat" page
    3. Customize system prompts on the "Settings" page
    
    ### Contact:
    
    For questions and suggestions about this application: example@example.com
    """)

# App info in sidebar
# st.sidebar.markdown("---")
# st.sidebar.caption(
#     "DocChat v1.0 | RAG-based Document Analysis System"
# )

# Create page navigation
pages = {
    "DocChat": [
        st.Page(main_page, title="Ana Sayfa", icon="📄"),
        st.Page(upload_document, title="Doküman Yükle", icon="📎"),
        st.Page(chat, title="Sohbet", icon="💬"),
        st.Page(settings, title="Ayarlar", icon="⚙️")
    ]
}

# Initialize navigation and run current page
current_page = st.navigation(pages, expanded=True)
current_page.run()

# Navigasyon bölümünden sonra footer bilgisini ekle
st.sidebar.markdown("<div style='position: fixed; bottom: 0; left: 0; padding: 10px; width: 100%; background-color: #f0f2f6;'><hr style='margin: 0;'><p style='font-size: 12px; color: #606060; text-align: center; margin: 0;'>DocChat v1.0 | RAG-based Document Analysis System</p></div>", unsafe_allow_html=True) 