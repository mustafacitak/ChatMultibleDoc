import streamlit as st
import os
from utils.config import init_app, DEFAULT_COLLECTION
from utils.vector_store import normalize_collection_name

# Initialize the app and perform necessary checks
app_ready = init_app()

# App title and configuration
st.set_page_config(
    page_title="DoChat - Document Analysis Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to increase navigation font size
st.markdown("""
<style>
    /* Increase navigation title font size */
    [data-testid="stSidebarNav"] p {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    /* Increase navigation menu item font size */
    .streamlit-nav {
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for collection name
if "collection_name" not in st.session_state:
    st.session_state.collection_name = DEFAULT_COLLECTION
    
# Initialize collection_name_set flag for dialog
if "collection_name_set" not in st.session_state:
    st.session_state.collection_name_set = False

# Initialize session state
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
    total_chunks = 0
    processed_files = []
    
    for idx, file in enumerate(files):
        # Process each file with spinner
        with st.spinner(f"Processing {file.name}..."):
            try:
                # Create a temp file to process
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(file.getvalue())
                    temp_path = temp_file.name
                    
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
                        # Add file name to successful list
                        processed_files.append(file.name)
                        total_chunks += len(docs)
                        st.success(f"Download successful: {file.name}")
                    else:
                        st.error(f"Failed to extract content from {file.name}")
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            finally:
                # Clean up the temp file
                try:
                    if 'temp_path' in locals():
                        os.unlink(temp_path)
                except:
                    pass
                    
    if total_chunks > 0:
        st.success(f"✅ Ready for chat! {len(files)} files processed into collection: {st.session_state.collection_name}")


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
                st.success(f"Download successful: {url}")
                
                # İşlem tamamlandı mesajı
                st.success(f"✅ Ready for chat! URL content processed into collection: {st.session_state.collection_name}")
            else:
                st.warning("No content could be extracted from the URL")
                
        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")

def process_uploaded_images(images):
    """Process multiple uploaded image files and PDFs with OCR"""
    import tempfile
    from utils.document_loader import load_image_ocr, process_pdf_with_ocr
    from utils.embeddings import get_local_embeddings, get_cached_embeddings
    from utils.vector_store import create_or_update_collection
    
    if not images:
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
    total_chunks = 0
    processed_files = []
    
    for idx, file in enumerate(images):
        # Get file extension
        file_ext = os.path.splitext(file.name)[1].lower()
        
        # Process file with spinner
        with st.spinner(f"Processing {file.name} with OCR..."):
            try:
                # Create a temp file to process
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    temp_file.write(file.getvalue())
                    temp_path = temp_file.name
                
                # Process based on file type
                if file_ext == '.pdf':
                    # Process PDF
                    docs = process_pdf_with_ocr(temp_path)
                else:
                    # Process image
                    docs = load_image_ocr(temp_path, file.name)
                
                if docs:
                    # Create or update vector store
                    create_or_update_collection(
                        docs, 
                        cached_embeddings, 
                        st.session_state.collection_name,
                        source_name=file.name
                    )
                    # Add file name to successful list
                    processed_files.append(file.name)
                    total_chunks += len(docs)
                    st.success(f"Download successful: {file.name}")
                else:
                    st.error(f"No text could be extracted from {file.name}")
            
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            
            finally:
                # Clean up the temp file
                try:
                    if 'temp_path' in locals():
                        os.unlink(temp_path)
                except:
                    pass
        
    if total_chunks > 0:
        st.success(f"✅ Ready for chat! {len(images)} files processed into collection: {st.session_state.collection_name}")

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

def get_existing_collections():
    """Get a list of existing collections from the db directory"""
    if not os.path.exists("db"):
        return []
    
    # Get all directories in the db folder - each directory is a collection
    collections = [d for d in os.listdir("db") if os.path.isdir(os.path.join("db", d))]
    return collections

# Navigation
def main_page():
    """Main page content - Application features and descriptions"""
    
    # Application description at the top instead of title
    st.markdown("""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f7ff;">
    <h2> 📄 Advanced AI Document Assistant</h2>
    <p>DocChat is the smartest way to interact with your documents! Upload a document, ask questions, get insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API key check
    if not app_ready:
        st.error("""
        **API key not found!** 
        
        Please create a `.env` file in the root directory and add your API key as follows:
        ```
        GOOGLE_API_KEY=your_api_key
        ```
        """)
        return
    
    # AI Capabilities - vertical layout
    st.subheader("AI Capabilities")
    st.markdown("""
    • Document Chat: Natural language conversation with your documents
    
    • Smart Task Classification: Automatically selects the best response type
    
    • Vector-Based Search: Semantic understanding for relevant answers
    
    • Source References: Trace answers back to source documents
    """)
    
    # Document Processing - vertical layout
    st.subheader("Document Processing")
    st.markdown("""
    • Multiple Document Types: PDF, Word, Excel, CSV, Text files
    
    • OCR Technology: Extract text from scanned PDFs and images
    
    • Web Content: Fetch and analyze content from URLs
    
    • Dynamic Chunking: Preserves context and meaning structure
    """)
    
    # Getting started
    st.subheader("Getting Started")
    
    st.markdown("""
    1. Upload Documents: Add files, web content, or use OCR for images in the 'Upload Document' page
    
    2. Set Collection Name: Organize your documents in named collections
    
    3. Configure Prompts (Optional): Select a system prompt or use automatic classification
    
    4. Start Chatting: Go to the 'Chat' page and ask questions about your documents
    """)
    
    # About and version
    st.subheader("About DocChat")
    
    st.markdown("""
    DocChat is a Retrieval-Augmented Generation (RAG) application powered by LangChain and Google Gemini.
    
    Version: 0.2
    
    Technologies:
    • LangChain: RAG architecture setup
    • Google Gemini API: AI model
    • ChromaDB: Vector database
    • Streamlit: User interface
    
    Features: Document processing, semantic search, intelligent chat, OCR
    
    [Upload Document](/Upload_Document) • [System Prompts](/System_Prompts) • [Chat](/Chat)
    """)

def upload_document():
    import sys
    import tempfile
    import uuid
    from urllib.parse import urlparse
    import time

    # Add parent directory to path to import utils
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.document_loader import load_document_from_file, load_document_from_url, load_image_ocr
    from utils.embeddings import get_local_embeddings, get_cached_embeddings
    from utils.vector_store import create_or_update_collection
    from utils.config import DEFAULT_COLLECTION

    # Initialize dialog flag if not exist
    if "show_collection_dialog" not in st.session_state:
        st.session_state.show_collection_dialog = False

    # Initialize collection_name if not exist
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = DEFAULT_COLLECTION
        st.session_state.collection_name_set = True
    
    # Initialize collection mode in session state
    if "collection_mode" not in st.session_state:
        st.session_state.collection_mode = "existing" if get_existing_collections() else "new"

    # Collection name dialog setup (when button is clicked)
    if st.session_state.show_collection_dialog:
        @st.dialog("Create New Collection", width="small")
        def create_collection_dialog():
            st.write("Enter a name for your new collection")
            name = st.text_input("Collection Name", value="", placeholder="my_collection", key="dialog_collection_name")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create", type="primary", key="dialog_save_btn"):
                    if not name.strip():
                        st.error("Collection name cannot be empty")
                        return
                    
                    normalized_name = normalize_collection_name(name)
                    st.session_state.collection_name = normalized_name  # Set the new collection as active
                    st.session_state.show_collection_dialog = False
                    st.session_state.collection_mode = "new"  # Set mode to new
                    
                    if normalized_name != name:
                        st.info(f"Collection name normalized: '{name}' -> '{normalized_name}'")
                    st.rerun()
            
            with col2:
                if st.button("Cancel", key="dialog_cancel_btn"):
                    st.session_state.show_collection_dialog = False
                    st.rerun()
        
        create_collection_dialog()

    # Page title
    st.title("Upload Document")
    
    # Collection selection or creation
    with st.sidebar:
        st.subheader("Collection Settings")
        
        # Get existing collections
        existing_collections = get_existing_collections()
        
        # Mode selector for collection management
        if existing_collections:
            # Let user choose between existing collections or creating a new one
            collection_options = ["Use Existing Collection", "Create New Collection"]
            selected_option = st.radio(
                "Collection Mode", 
                options=collection_options,
                index=0 if st.session_state.collection_mode == "existing" else 1,
                key="collection_mode_radio"
            )
            
            # Update the collection mode based on selection
            st.session_state.collection_mode = "existing" if selected_option == "Use Existing Collection" else "new"
            
            # Show appropriate UI based on selected mode
            if st.session_state.collection_mode == "existing":
                # Existing collections dropdown
                st.write("Select an existing collection:")
                selected_collection = st.selectbox(
                    "Available Collections",
                    options=existing_collections,
                    index=existing_collections.index(st.session_state.collection_name) if st.session_state.collection_name in existing_collections else 0,
                    key="existing_collection_select"
                )
                
                # Update the active collection when user selects from dropdown
                if selected_collection != st.session_state.collection_name:
                    st.session_state.collection_name = selected_collection
                    st.info(f"Active collection changed to: {selected_collection}")
            else:
                # New collection creation
                st.write("Create a new collection:")
                if st.button("Add New Collection", key="add_collection_btn", type="primary"):
                    st.session_state.show_collection_dialog = True
                    st.rerun()
        else:
            # No existing collections, only show new collection creation
            st.write("No collections found. Create your first collection:")
            if st.button("Create Collection", key="create_first_collection_btn", type="primary"):
                st.session_state.show_collection_dialog = True
                st.rerun()
        
        # Show active collection status
        st.divider()
        st.subheader("Active Collection")
        
        # Format the display based on whether it's new or existing
        if st.session_state.collection_name in existing_collections:
            st.success(f"{st.session_state.collection_name}")
            st.caption("Existing collection")
        else:
            st.info(f"{st.session_state.collection_name}")
            st.caption("New collection will be created when documents are processed")
    
    # Create tabs for different upload methods
    tab1, tab2, tab3 = st.tabs(["📎 Upload Files", "🔗 Add URL", "🖼️ Image OCR"])
    
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
            if st.button("Process Files", type="primary", key="process_files_btn"):
                process_uploaded_files(uploaded_files)
    
    # URL content tab
    with tab2:
        st.subheader("Add Content from URL")
        
        # URL input field
        url = st.text_input("Enter URL", placeholder="https://example.com", key="url_input")
        
        # Process URL content
        if url:
            if st.button("Fetch and Process URL", type="primary", key="fetch_url_btn"):
                process_url_content(url)
    
    # Image OCR tab
    with tab3:
        st.subheader("Image OCR")
        
        # Supported image types explanation
        st.markdown("""
        Upload images or scanned PDFs for OCR text extraction:
        - 🖼️ **JPG/JPEG** - JPEG images
        - 🖼️ **PNG** - PNG images
        - 🖼️ **TIFF** - TIFF format
        - 🖼️ **BMP** - Bitmap images
        - 📄 **PDF** - Scanned PDF documents
        
        The system will extract text using OCR technology.
        """)
        
        # Image uploader
        uploaded_images = st.file_uploader(
            "Choose images or scanned PDFs to process",
            type=["jpg", "jpeg", "png", "tiff", "tif", "bmp", "pdf"],
            accept_multiple_files=True
        )
        
        # Process uploaded images
        if uploaded_images:
            if st.button("Process with OCR", type="primary", key="process_ocr_btn"):
                process_uploaded_images(uploaded_images)

def system_prompts():
    """System Prompts page - Prompt management and selection"""
    import sys
    import yaml

    # Import utility modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.langchain_helpers import load_system_prompts

    # Title
    st.title("System Prompts")

    # System prompt file path
    system_prompt_path = "config/system_prompt.yaml"
    
    # Load system prompts
    try:
        prompts = load_system_prompts(system_prompt_path)
        
        # Title and description
        st.markdown("""
        System prompts define the AI model's response style and format.
        You can use different prompts for different tasks.
        """)
        
        # Intelligent task classification feature
        
        # Initialize session state
        if "use_smart_classification" not in st.session_state:
            st.session_state.use_smart_classification = True
            
        if "active_prompt" not in st.session_state:
            st.session_state.active_prompt = "default_chat"
        
        # Mode selection
        col1, col2 = st.columns(2)
        with col1:
            smart_mode = st.toggle("Intelligent Task Classification", 
                                  value=st.session_state.use_smart_classification,
                                  help="When active, user queries are automatically classified and appropriate prompts are selected")
        
        # Update session state
        st.session_state.use_smart_classification = smart_mode
        
        # Manual prompt selection (when smart mode is off)
        if not smart_mode:
            with col2:
                prompt_list = list(prompts.keys())
                active_prompt = st.selectbox(
                    "Active System Prompt",
                    prompt_list,
                    index=prompt_list.index(st.session_state.active_prompt) if st.session_state.active_prompt in prompt_list else 0
                )
                st.session_state.active_prompt = active_prompt
        else:
            st.info("✨ **Intelligent Task Classification is active!** User queries will be automatically analyzed and the most appropriate prompt will be selected.")
        
        # Prompt preview
        st.subheader("Available Prompts", anchor=False)
        
        # Show prompts as editable panels
        for prompt_name, prompt_text in prompts.items():
            with st.expander(f"{prompt_name}"):
                # Prompt type description
                if prompt_name == "default_chat":
                    st.caption("Prompt used for general conversation.")
                elif prompt_name == "document_analysis":
                    st.caption("Prompt used for document analysis.")
                elif prompt_name == "question_answering":
                    st.caption("Prompt used for question answering.")
                elif prompt_name == "code_explanation":
                    st.caption("Prompt used for code explanation.")
                elif prompt_name == "summarization":
                    st.caption("Prompt used for summarization.")
                elif prompt_name == "data_analysis":
                    st.caption("Prompt used for data analysis.")
                elif prompt_name == "web_content_summary":
                    st.caption("Prompt used for web content summarization.")
                elif prompt_name == "decision_support":
                    st.caption("Prompt used for decision support.")
                elif prompt_name == "information_retrieval":
                    st.caption("Prompt used for information retrieval.")
                elif prompt_name == "error_detection":
                    st.caption("Prompt used for error detection.")
                
                # Editing area
                st.text_area(
                    f"{prompt_name} Content", 
                    value=prompt_text, 
                    height=200,
                    key=f"prompt_{prompt_name}"
                )
                
                # Delete button
                if prompt_name not in ["default_chat", "document_analysis", "question_answering", "code_explanation", "summarization"]:
                    if st.button(f"Delete this prompt", key=f"delete_{prompt_name}"):
                        # Delete prompt
                        prompts.pop(prompt_name)
                        
                        # Save to YAML file
                        with open(system_prompt_path, 'w', encoding='utf-8') as file:
                            yaml.dump(prompts, file, default_flow_style=False, allow_unicode=True)
                        
                        st.success(f"Prompt '{prompt_name}' deleted!")
                        st.rerun()
                else:
                    st.caption("Core prompts cannot be deleted.")
        
        # Add new prompt
        st.subheader("Add New Prompt", anchor=False)
        
        # Form for adding new prompt
        with st.form(key="add_prompt_form"):
            new_prompt_name = st.text_input("New Prompt Name", 
                                          placeholder="my_custom_prompt",
                                          help="Prompt name should be unique and lowercase, use underscores instead of spaces")
            
            new_prompt_text = st.text_area("Prompt Content", 
                                         placeholder="You are an expert in...\n\n* Respond in this way...\n* Analyze like this...",
                                         height=150)
            
            submit_button = st.form_submit_button(label="Add Prompt", type="primary")
            
            if submit_button:
                if not new_prompt_name or not new_prompt_text:
                    st.error("Prompt name and content cannot be empty!")
                elif new_prompt_name in prompts:
                    st.error(f"A prompt named '{new_prompt_name}' already exists! Please use a different name.")
                else:
                    # Add new prompt
                    prompts[new_prompt_name] = new_prompt_text
                    
                    # Save to YAML file
                    with open(system_prompt_path, 'w', encoding='utf-8') as file:
                        yaml.dump(prompts, file, default_flow_style=False, allow_unicode=True)
                    
                    st.success(f"Prompt '{new_prompt_name}' added successfully!")
                    st.rerun()
        
        # Update prompts
        st.subheader("Save Changes", anchor=False)
        
        # Save all prompt changes
        if st.button("Save All Changes", type="primary", key="save_all_prompts"):
            # Collect all updated prompts
            updated_prompts = {}
            for prompt_name in prompts.keys():
                updated_text = st.session_state.get(f"prompt_{prompt_name}")
                if updated_text:
                    updated_prompts[prompt_name] = updated_text
            
            # Save to YAML file
            with open(system_prompt_path, 'w', encoding='utf-8') as file:
                yaml.dump(updated_prompts, file, default_flow_style=False, allow_unicode=True)
            
            st.success("All prompt changes saved!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
        # Option to create default prompts
        if st.button("Create Default Prompts", key="create_default_prompts_btn"):
            create_default_prompts(system_prompt_path)
            st.success("Default prompts created!")
            st.rerun()

def chat():
    import sys
    import time

    # Import utility modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.embeddings import get_local_embeddings, get_cached_embeddings
    from utils.rag import load_collection, get_qa_chain, create_weighted_retriever
    from utils.langchain_helpers import get_llm, load_system_prompts, format_sources
    from utils.config import DEFAULT_COLLECTION, MODEL_NAME
    from utils.vector_store import normalize_collection_name
    from utils.task_classifier import get_task_based_rag, get_qa_llm

    # Title
    st.title("🗯️ Ask Your AI Assistant")

    # Response stream generator function
    def response_generator(result):
        words = result.split()
        for word in words:
            yield word + " "
            time.sleep(0.03)  # Increased flow speed - for slower animation

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Cache for model and embeddings
    if "cached_embed_model" not in st.session_state:
        st.session_state.cached_embed_model = None
    
    if "cached_llm" not in st.session_state:
        st.session_state.cached_llm = None

    # Normalize collection name (if needed)
    st.session_state.collection_name = normalize_collection_name(st.session_state.collection_name)

    # Collection info
    st.sidebar.info(f"Active Collection: **{st.session_state.collection_name}**")
    st.sidebar.caption(f"Using model: **{MODEL_NAME}**")
    
    # Clear chat history button
    if st.sidebar.button("Clear Chat History", key="clear_chat_btn"):
        st.session_state.messages = []
        st.rerun()

    # Display previous messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources in assistant messages
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    st.write(message["sources"])

    # User input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            # Show thinking indicator and spinner side by side
            thinking_container = st.empty()
            
            # Custom HTML with spinner and "Thinking..." text side by side
            thinking_container.markdown("""
            <div style="display: flex; align-items: center; gap: 8px;">
                <div class="spinner">
                    <style>
                    .spinner {
                        width: 18px;
                        height: 18px;
                        border: 3px solid rgba(0, 0, 0, 0.1);
                        border-radius: 50%;
                        border-top-color: #2196F3;
                        animation: spin 1s ease-in-out infinite;
                    }
                    @keyframes spin {
                        to { transform: rotate(360deg); }
                    }
                    </style>
                </div>
                <span><i>Thinking...</i></span>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Load embedding model from cache or create new
                if st.session_state.cached_embed_model is None:
                    embed_model = get_local_embeddings()
                    cached_embeddings = get_cached_embeddings(embed_model)
                    st.session_state.cached_embed_model = cached_embeddings
                else:
                    cached_embeddings = st.session_state.cached_embed_model
                
                try:
                    # Load collection
                    vectordb = load_collection(st.session_state.collection_name, cached_embeddings)
                    
                    # Check if Intelligent Task Classification is enabled
                    if "use_smart_classification" in st.session_state and st.session_state.use_smart_classification:
                        # Use intelligent classification
                        # First get a sample document to provide context
                        sample_retriever = vectordb.as_retriever(search_kwargs={"k": 1})
                        sample_docs = sample_retriever.get_relevant_documents(prompt)
                        doc_sample = sample_docs[0].page_content if sample_docs else None
                        
                        # Task classification and query improvement
                        task_type, improved_query = get_task_based_rag(
                            prompt, 
                            st.session_state.messages, 
                            doc_sample
                        )
                    else:
                        # Use manually selected prompt
                        task_type = st.session_state.active_prompt
                        improved_query = prompt
                    
                    # Load system prompts
                    system_prompts = load_system_prompts()
                    
                    # Select system prompt based on classified task type
                    system_prompt = system_prompts.get(task_type,
                        "Answer questions based on the information from the user's documents.")
                    
                    # Create weighted retriever for query
                    retriever = create_weighted_retriever(vectordb, search_kwargs={"k": 4})
                    
                    # Load LLM model from cache or create new
                    if st.session_state.cached_llm is None:
                        # Soru-cevap için optimize edilmiş LLM'i kullan
                        llm = get_qa_llm()
                        st.session_state.cached_llm = llm
                    else:
                        llm = st.session_state.cached_llm
                    
                    try:
                        # Create QA chain
                        qa_chain = get_qa_chain(llm, retriever, system_prompt)
                        
                        # İyileştirilmiş sorguyu kullan
                        result = qa_chain({"query": improved_query})
                        answer = result["result"]
                        source_docs = result["source_documents"]
                    except Exception as chain_error:
                        # Fallback: Using direct approach instead of LangChain API
                        
                        try:
                            # Get documents directly from retriever
                            relevant_docs = retriever.get_relevant_documents(improved_query)
                            
                            # Create prompt template
                            docs_text = ""
                            for i, doc in enumerate(relevant_docs):
                                docs_text += f"\nDocument {i+1}:\n{doc.page_content}\n"
                            
                            prompt_template = f"""
                            {system_prompt}
                            
                            Use the following text to answer the question:
                            
                            {docs_text}
                            
                            Question: {improved_query}
                            Answer:
                            """
                            
                            # Query LLM directly
                            answer = llm.invoke(prompt_template).content
                            source_docs = relevant_docs
                        except Exception as fallback_error:
                            raise Exception(f"Both approaches failed. Original error: {chain_error}. Fallback error: {fallback_error}")
                    
                    # Format sources
                    sources_text = format_sources(source_docs)
                    
                    # Clear thinking indicator
                    thinking_container.empty()
                    
                    # Display response as stream
                    response = st.write_stream(response_generator(answer))
                    
                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources_text
                    })
                    
                    # Show sources
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
                    st.error(f"Error loading collection: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Error: {str(e)}",
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

# App info in sidebar
# st.sidebar.markdown("---")
# st.sidebar.caption(
#     "DoChat v1.0 | RAG-based Document Analysis System"
# )

# Create page navigation
pages = {
    "DoChat": [
        st.Page(main_page, title="Home", icon="📄"),
        st.Page(upload_document, title="Upload Document", icon="📎"),
        st.Page(system_prompts, title="System Prompts", icon="🔧"),
        st.Page(chat, title="Chat", icon="💬")
    ]
}

# Initialize navigation and run current page
current_page = st.navigation(pages, expanded=True)
current_page.run()

# Add footer information after navigation
st.sidebar.markdown("<div style='position: fixed; bottom: 0; left: 0; padding: 10px; width: 100%; background-color: #f0f2f6;'><hr style='margin: 0;'><p style='font-size: 12px; color: #606060; text-align: center; margin: 0;'>DocChat v1.0 | RAG-based Document Analysis System</p></div>", unsafe_allow_html=True) 