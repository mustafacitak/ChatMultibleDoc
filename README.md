# DoChat v0.3

## 1. Code Architecture

### 1.1. Folder and File Structure

```
.
├── main.py
├── requirements.txt
├── config/
│   ├── chunking_config.yaml
│   ├── task_classification.yaml
│   └── prompt_improve_query.yaml
├── db/
│   ├── cache/
│   └── collections/
│       ├── <collection_name>/
│           ├── memory.jsonl
│           ├── documents.jsonl
│           ├── sources.txt
│           └── config.json
├── pages/
│   ├── 1_upload_document.py
│   ├── 2_chat.py
├── utils/
│   ├── binary_quantization.py
│   ├── embedding_utils.py
│   ├── funnel_rag.py
│   ├── adaptive_rag.py
│   ├── cache_utils.py
│   ├── rcot_reflection.py
│   ├── chunking_utils.py
│   ├── query_enrichment.py
│   ├── hash_utils.py
│   ├── duplicate_utils.py
│   ├── net_search.py
│   ├── collection_utils.py
│   └── task_classifier.py
├── .gitignore
├── venv/
└── (various temp and log files)
```

### 1.2. Folder and File Roles

- **main.py**: Application entry point. Provides multi-page navigation with Streamlit.
- **requirements.txt**: Lists all Python libraries used in the project.
- **config/**: Contains application and model settings in YAML format.
- **db/**: Data folder where documents and caches are stored.
  - **cache/**: RAG cache files (embedding, retrieval, context, response).
  - **collections/**: Separate folder for each collection, contains documents and metadata.
- **pages/**: Streamlit pages (document upload and chat interface).
- **utils/**: All helper functions, data processing, embedding, chunking, caching, RAG, LLM integration, error management, etc.
- **venv/**: Python virtual environment.
- **.gitignore**: Files to be excluded from version control.

### 1.3. Architectural Approach

- **Service-based and modular structure**: Each functional area (chunking, embedding, cache, RAG, LLM integration) is in separate modules.
- **Streamlit-based frontend**: User interface and interactions are managed with Streamlit.
- **Event-driven flow**: User interactions (document upload, chat) are triggered by events.
- **Config-driven**: All important parameters are read from YAML files.
- **RAG (Retrieval-Augmented Generation)**: Information retrieval from documents and answer generation with LLM.

### 1.4. Language, Framework, and Libraries Used

- **Language**: Python 3
- **Main Framework**: Streamlit (web interface)
- **LLM and Embedding**: sentence-transformers, google-generativeai, LangChain
- **Data Processing**: pandas, numpy, PyPDF2, python-docx, paddleocr, PIL, PyMuPDF
- **Helpers**: loguru (logging), python-dotenv (env management), pyyaml (yaml reading), requests, aiohttp, bs4a
- **Search and Vector DB**: typesense, meilisearch, opensearch-py, pyarrow

## User Interface Screenshots

### Upload Document Interface
![Upload Document Interface](Upload.png)

### Chat Interface
![Chat Interface](chat.png)

### RAG System Optimization Settings
![RAG Optimization Settings](RAGtype.png)

### System Flow Diagram
![DoChat System Flow Diagram](diagram.png)

---

## 2. Module Flow

### 2.1. User Scenario and Application Flow

#### 2.1.1. Document Upload (pages/1_upload_document.py)

1. The user uploads a document via the interface (PDF, DOCX, TXT, CSV, Excel, image, or URL).
2. The appropriate text extractor function is called according to the file type (`extract_text_from_pdf`, `extract_text_from_docx`, etc.).
3. The extracted text is split into chunks (`dynamic_chunking`, `sentence_chunking`).
4. An embedding is generated for each chunk (`generate_embedding`).
5. Chunks and embeddings are saved to the collection folder (`save_chunks_to_jsonl`).
6. Name normalization and management between collections is performed (`normalize_collection_name`, `get_existing_collections`).
7. If an error occurs during upload and processing, the user is notified (with the error_handler decorator).

#### 2.1.2. Chat (pages/2_chat.py)

1. The user writes and submits a question.
2. Query preprocessing and enrichment are performed (`improve_query`, `preprocess_query`).
3. The query type is classified (`classify_query_type`).
4. Query embedding is generated (`generate_embedding`).
5. RAG retrieval is performed:
   - Cache is checked first (`get_cached_retrieval`).
   - If not found, the most relevant chunks are retrieved from the collection (`get_top_k_chunks`, `FunnelRAG`, `Adaptive-RAG`).
6. Context is prepared (`prepare_context_for_llm`).
7. An answer is generated with the LLM (`run_rcot_reflection`, `generate_initial_response`).
8. The answer is shown to the user and feedback can be collected.
9. All steps are logged and the user is notified if an error occurs.

#### 2.1.3. Important Middleware/Event Handlers

- `error_handler` / `async_error_handler`: Error catching and user notification in all critical functions.
- Session management: Collection and user-specific information is stored with `st.session_state`.
- Cache management: Multi-level cache for embedding, retrieval, context, and response with `cache_utils`.

---

## 3. Function and Class Reviews

### 3.1. extract_text_from_pdf (pages/1_upload_document.py)

- **Definition**: Extracts text from a PDF file.
- **Input Parameters**:
  - `file_path` (str): Path to the PDF file.
- **Return Value**: (str) Extracted text.
- **Usage**: In the document upload flow, when a PDF file is uploaded.
- **Purpose**: To process the content of the uploaded PDF file as text.
- **Key Operations**: Page-by-page text extraction with PyPDF2, error logging.
- **Side Effects**: User notification in case of error.
- **Default Values**: None.
- **Example Usage**:
  ```python
  text = extract_text_from_pdf("/path/to/file.pdf")
  ```

### 3.2. generate_embedding (utils/embedding_utils.py)

- **Definition**: Generates an embedding vector for the given text, checks the cache.
- **Input Parameters**:
  - `text` (str): Text to generate embedding for.
- **Return Value**: (list[float]) Embedding vector.
- **Usage**: After chunking, during query processing.
- **Purpose**: To represent texts in vector space and calculate similarity.
- **Key Operations**: Possible cache check, binary quantization support, embedding generation.
- **Side Effects**: Written to embedding cache.
- **Default Values**: None.
- **Example Usage**:
  ```python
  emb = generate_embedding("example text")
  ```

### 3.3. get_config_param (utils/embedding_utils.py)

- **Definition**: Reads a parameter from the YAML config file.
- **Input Parameters**:
  - `param_path` (str): Parameter path in dot notation.
  - `default_value` (any): Default value.
  - `config_path` (str, optional): Config file path.
- **Return Value**: (any) Parameter value.
- **Usage**: In all modules, for parameter reading operations.
- **Purpose**: To make the code flexible and independent of hardcoded parameters.
- **Key Operations**: YAML reading, error logging.
- **Side Effects**: None.
- **Default Values**: Yes.
- **Example Usage**:
  ```python
  top_k = get_config_param('chunking.top_k', 5)
  ```

### 3.4. error_handler (pages/1_upload_document.py, pages/2_chat.py)

- **Definition**: Decorator that wraps functions and catches errors.
- **Input Parameters**: Parameters of the wrapped function.
- **Return Value**: Original return of the function or None.
- **Usage**: All critical functions.
- **Purpose**: To show user-friendly error messages and log them.
- **Key Operations**: try/except, logging, Streamlit error notification.
- **Side Effects**: Shows error message to the user.
- **Default Values**: None.

---

## 4. Modules and Connections

### 4.1. Inter-Module Usage

- **pages/1_upload_document.py**: 
  - `utils.hash_utils`, `utils.collection_utils`, `utils.chunking_utils`, `utils.embedding_utils`, `utils.duplicate_utils`
- **pages/2_chat.py**: 
  - `utils.embedding_utils`, `utils.query_enrichment`, `utils.task_classifier`, `utils.rcot_reflection`, `utils.hash_utils`, `utils.adaptive_rag`, `utils.funnel_rag`, `utils.cache_utils`
- **utils/embedding_utils.py**: 
  - `sentence-transformers`, `utils.cache_utils`, `utils.binary_quantization`
- **utils/cache_utils.py**: 
  - `utils.embedding_utils`, `numpy`, `loguru`
- **utils/rcot_reflection.py**: 
  - `google.generativeai`, `utils.embedding_utils`
- **utils/query_enrichment.py**: 
  - `google.generativeai`, `utils/embedding_utils`

### 4.2. Dependency Management

- All dependencies are managed with `requirements.txt`.
- Environment variables are managed with `.env` file.
- Inter-module imports are handled directly with Python import system.

### 4.3. Data Transfer Between Modules

- Through function calls and parameters.
- Between collections via file system (jsonl, txt).
- Fast data sharing between modules with caches (`cache_utils`).

### 4.4. Separation of Concerns

- **pages/**: Only UI and user interaction.
- **utils/**: All business logic, data processing, LLM, embedding, cache, chunking, etc.
- **db/**: Persistent data and cache.
- **config/**: Parameter and settings management.

---

## 5. Common Utility Functions (Utils, Helpers)

- `generate_embedding`: All embedding operations.
- `get_config_param`: All parameter reading operations.
- `error_handler`: Error management.
- `dynamic_chunking`, `sentence_chunking`: Text splitting operations.
- `is_duplicate_by_embedding`: Duplicate record check.
- `cache_utils` functions: get_cached_embedding, cache_embedding, get_cached_retrieval, cache_retrieval, etc.

Each is used multiple times and for different purposes in the relevant modules.

---

## 6. Configuration and Environment Variables

### 6.1. .env and Environment Variables

- `GOOGLE_API_KEY` / `GEMINI_API_KEY`: For LLM and Google API access.
- **Usage**: Read with `os.getenv`, priority order is set.

### 6.2. config/chunking_config.yaml

- `chunking.top_k`: Number of chunks to retrieve for retrieval.
- `binary_quantization.enabled`: Is binary quantization active?
- `cached_rag.enabled`: Is cache active?
- `gemini.model_name`: LLM model to use.
- `app.default_collection`: Default collection name.
- **Others**: All important parameters are here.

---

## 7. External Services and Integrations

- **Google Gemini API**: For LLM answers.
- **LangChain**: Document upload and processing.
- **Typesense, Meilisearch, OpenSearch**: Vector search and retrieval (optional).
- **requests, aiohttp**: Web and API calls.

---

## 8. Error Management

- `error_handler` / `async_error_handler`: In all critical functions.
- **Logging**: With loguru, both to terminal and file.
- **User Notification**: Streamlit error message.
- **Global error handler**: Per function, no central global handler.

---

## 9. Tests

- No test file is visible. (If there is, specify and I will detail it.)
- If tests are added, the modular structure is suitable for unit and integration tests.

---

## 10. Security

- **API Key privacy**: Protected with .env and .gitignore.
- **Input validation**: File type and content check exists.
- **Encryption**: None (should be added if needed).
- **Potential vulnerabilities**: Rate limit and input sanitization are important in file upload and LLM calls.

---

## 1.5. Technologies and Components Used

- **Python 3**  
  Main development language.
- **Streamlit**  
  Web-based user interface and interaction.
- **sentence-transformers**  
  Text embedding (vectorization) operations.
- **google-generativeai**  
  LLM-based answer generation and query enrichment.
- **LangChain**  
  Document upload and processing (especially for file types like Excel, CSV).
- **pandas, numpy**  
  Data processing and analysis.
- **PyPDF2, python-docx, paddleocr, PIL, PyMuPDF**  
  Text extraction from different document and image types.
- **loguru**  
  Advanced logging.
- **pyyaml, python-dotenv**  
  Configuration and environment variable management.
- **requests, aiohttp, bs4**  
  Web search and access to external data sources.
- **typesense, meilisearch, opensearch-py, pyarrow**  
  (Optional) Vector-based search and retrieval infrastructure.

---

## 2. Main Systems and Modules

- **Query enrichment** (`improve_query`)
  - Improves user queries in terms of grammar and meaning, LLM-supported.
- **Task classification** (`classify_query_type`)
  - Automatically determines the type of query such as summary, analysis, general information.
- **Hash mechanism** (`chunk/file hash`)
  - Uniqueness and processed check at chunk and file level.
- **Embedding-based duplicate check**
  - Prevents re-adding of same or very similar content.
- **chunking_utils** (Dynamic chunking)
  - Splits texts into dynamic parts based on sentence or word.
- **Collection management**
  - Storing, adding, and updating documents in collections.
- **Web search / net_search**
  - Asynchronous search from sources like DuckDuckGo, arXiv, OpenAlex, Typesense, Meilisearch, OpenSearch.
- **RAG and retrieval**
  - Information retrieval from documents and answer generation with LLM (Retrieval-Augmented Generation).
- **Self-reflection RAG and Reflective Chain of Thought (RCoT)**
  - Step-by-step reasoning structure that checks and, if necessary, updates and verifies LLM answers. (See: `rcot_reflection.py`)

---

## 3. Optimization Systems

### 3.1. Cached RAG
- Multi-layered caching at embedding, retrieval, context, and response levels.
- Supports both exact match and semantic (similarity-based) cache.

### 3.2. Adaptive-RAG
- Dynamically adjusts chunk count and context size according to query complexity.
- Uses less data for simple questions, more for complex ones.

### 3.3. FunnelRAG (Two-Stage Retrieval)
- Fast and coarse filtering in the first stage (TF-IDF, keyword), precise ranking with embedding in the second stage.
- Increases efficiency and quality in large collections.

### 3.4. Binary Quantization
- Converts embedding vectors to binary form to speed up similarity calculations.
- Hybrid mode provides both speed and accuracy optimization.

---

## 4. Reranking System (Assumed Scenario)

**Reranking** is the process of re-scoring the final chunk list to be presented to the user with a more precise model after retrieval. In the project, the second stage of FunnelRAG is an example of reranking, but a more advanced reranking system could be integrated as follows:

- **Where?**
  - In `utils/funnel_rag.py` or `utils/embedding_utils.py`, after the `get_top_k_chunks` function.
  - Or after retrieval, the top 10-20 chunks could be re-scored with an LLM-based reranker (e.g., a transformer model) and the best 5 selected.

- **How?**
  - Each chunk is given to a reranker model (e.g., cross-encoder, LLM) together with the query.
  - The model produces a score for each (query, chunk) pair.
  - Chunks are re-ranked according to this score and the highest scoring ones are selected.
  - Example in code:
    ```python
    # Top 20 chunks after retrieval
    candidates = top_chunks[:20]
    rerank_scores = [llm_reranker(query, c['text']) for c in candidates]
    reranked = [c for _, c in sorted(zip(rerank_scores, candidates), reverse=True)]
    final_chunks = reranked[:5]
    ```
  - This structure is especially used to highlight the most meaningful one among very similar chunks.

- **Place in the Flow:**
  1. User query is received.
  2. Query is improved with `improve_query`.
  3. Query type is determined with `classify_query_type`.
  4. Top 20 chunks are selected with retrieval.
  5. **Reranking**: These chunks are re-scored with an LLM-based reranker, and the best 5 are selected.
  6. Selected chunks are given to the LLM as context.
  7. LLM answer is generated, checked step by step with RCoT, and updated if necessary.
  8. The answer is presented to the user.

- **Benefit:**
  - Provides the user with the most relevant and meaningful content.
  - Increases quality especially in large collections.

---

## 5. Advanced Methods

- **Self-reflection RAG and Reflective Chain of Thought (RCoT)**
  - Step-by-step reasoning structure that checks and, if necessary, updates and verifies LLM answers.
  - Implemented in the `rcot_reflection.py` module in the project.

---

## 6. Other Important Components

- **Query enrichment** (improve_query)
- **Task classification** (classify_query_type)
- **Hash mechanism** (chunk/file hash)
- **Embedding-based duplicate check**
- **chunking_utils** (dynamic chunking)
- **Collection management**
- **Web search/net_search**
- **RAG and retrieval**

---

## 7. Example Flow (Including Reranking)

1. User enters a query from the interface.
2. Query is improved with `improve_query`.
3. Query type is determined with `classify_query_type`.
4. Top 20 chunks are selected with retrieval.
5. **Reranking**: These chunks are re-scored with an LLM-based reranker, and the best 5 are selected.
6. Selected chunks are given to the LLM as context.
7. LLM answer is generated, checked step by step with RCoT, and updated if necessary.
8. The answer is presented to the user.

---

## Contribution and License

This project provides a sample infrastructure for those who want to develop a modular and extensible RAG-based document chat system. To contribute, please open an issue or send a pull request. 
