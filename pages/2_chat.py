import os
import functools
import sys
import asyncio
from dotenv import load_dotenv
import streamlit as st
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("terminal_log.txt", rotation="5 MB", level="DEBUG")

# .env dosyasını yükle
load_dotenv()

GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.embedding_utils import generate_embedding, get_top_k_chunks, prepare_context_for_llm, get_config_param
from utils.query_enrichment import preprocess_query, improve_query, get_model_name_from_config
from utils.task_classifier import classify_query_type
from utils.rcot_reflection import (
    generate_initial_response, breakdown_reasoning_steps, validate_response_accuracy, 
    update_response, save_to_memory_log, run_token_optimized_rcot
)
from utils.hash_utils import load_chunks_from_jsonl

# Optimization modules import
try:
    from utils.adaptive_rag import analyze_query_complexity, get_adaptive_chunk_count, get_adaptive_context_size
    ADAPTIVE_RAG_ENABLED = True
    logger.info("Adaptive-RAG system enabled.")
except ImportError:
    ADAPTIVE_RAG_ENABLED = False
    logger.warning("Adaptive-RAG module not found, feature disabled.")

try:
    from utils.funnel_rag import FunnelRAG
    funnel_rag_instance = FunnelRAG()
    FUNNEL_RAG_ENABLED = True
    logger.info("FUNNELRAG (Two-Stage Retrieval) system enabled.")
except ImportError:
    FUNNEL_RAG_ENABLED = False
    logger.warning("FUNNELRAG module not found, feature disabled.")

try:
    from utils.binary_quantization import get_binary_quantizer
    binary_quantizer = get_binary_quantizer()
    BINARY_QUANTIZATION_ENABLED = True
    logger.info("Binary Quantization system enabled.")
except ImportError:
    BINARY_QUANTIZATION_ENABLED = False
    logger.warning("Binary Quantization module not found, feature disabled.")

# Cached RAG integration
try:
    from utils.cache_utils import (
        get_cached_embedding, cache_embedding,
        get_cached_retrieval, cache_retrieval,
        get_cached_context, cache_context,
        get_cached_response, cache_response,
        get_cache_stats, clear_cache, init_cache
    )
    CACHED_RAG_ENABLED = True
    logger.info("Cached RAG system enabled.")
except ImportError:
    CACHED_RAG_ENABLED = False
    logger.warning("Cached RAG module not found, caching disabled.")

# Error handler decorators
# ... (error_handler and async_error_handler functions will be added here)
# ... (process_query, run_rcot_reflection, chat_stream, save_feedback functions will be added here)
# ... (chat function and UI code will be added here)

# The rest of the code remains the same, only UI strings are translated below

def error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error occurred - Function: {func.__name__}, Error: {e}")
            st.error(f"An error occurred during the operation: {e}")
            return None
    return wrapper

def async_error_handler(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Async error occurred - Function: {func.__name__}, Error: {e}")
            st.error(f"An error occurred during the operation: {e}")
            return None
    return wrapper

from utils.collection_utils import get_existing_collections
from utils.hash_utils import get_file_hash
import json

COLLECTION_PATH_BASE = "db/collections"
DEFAULT_COLLECTION = "ornek_koleksiyon"
MODEL_NAME = get_model_name_from_config()
CONFIG_PATH = None
DOCS_PATH = None
MEMORY_PATH = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = DEFAULT_COLLECTION

def save_feedback(index):
    st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]

if "history" not in st.session_state:
    st.session_state.history = []

def chat():
    st.title("💬 Ask a Question")
    with st.sidebar:
        st.info(f"Active Collection: **{st.session_state.collection_name}**")
        st.caption(f"Model in use: **{MODEL_NAME}**")
        
        if st.button("Clear Chat History", key="clear_chat_btn"):
            st.session_state.history = []
            st.rerun()
    for i, message in enumerate(st.session_state.history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                feedback = message.get("feedback", None)
                st.session_state[f"feedback_{i}"] = feedback
                st.feedback(
                    "thumbs",
                    key=f"feedback_{i}",
                    disabled=feedback is not None,
                    on_change=save_feedback,
                    args=[i],
                )
                if "sources" in message and message["sources"]:
                    with st.expander("Sources"):
                        st.write(message["sources"])
    if prompt := st.chat_input("Type your question..."):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            thinking_container = st.empty()
            thinking_container.markdown("""
            <div style='display: flex; align-items: center; gap: 8px;'>
                <div class='spinner'>
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
            yanit, sources = asyncio.run(chat_stream(prompt))
            thinking_container.empty()
            response = st.write(yanit)
            st.feedback(
                "thumbs",
                key=f"feedback_{len(st.session_state.history)}",
                on_change=save_feedback,
                args=[len(st.session_state.history)],
            )
            if sources:
                with st.expander("Sources"):
                    st.write(sources)
        st.session_state.history.append({"role": "assistant", "content": yanit, "sources": sources})

@error_handler
async def chat_stream(prompt):
    collection_name = st.session_state.collection_name
    docs_path = f"db/collections/{collection_name}/documents.jsonl"
    gemini_api_key = GEMINI_API_KEY

    # Sorgu embedding'i oluştur
    query_embedding = generate_embedding(prompt)
    # Chunk'ları yükle
    chunks = load_chunks_from_jsonl(docs_path)
    # En alakalı chunk'ları getir
    top_chunks = get_top_k_chunks(query_embedding, chunks, k=5)
    # Bağlamı hazırla
    context = prepare_context_for_llm(top_chunks, prompt)
    # LLM'den yanıt al
    yanit = generate_initial_response(prompt, context, gemini_api_key)
    # Kaynakları ekle
    sources = [chunk.get('source', '') for chunk in top_chunks]
    return yanit, sources

chat()