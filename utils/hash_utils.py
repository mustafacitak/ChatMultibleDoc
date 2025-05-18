import os
import json
import hashlib
from loguru import logger

def calculate_chunk_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_file_hash(file_path: str) -> str:
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except Exception as e:
        logger.error(f"Hash hesaplama hatası: {e}")
        return ""

def is_file_already_processed(file_hash: str, jsonl_path: str) -> bool:
    if not os.path.exists(jsonl_path):
        return False
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get('file_hash') == file_hash:
                        logger.warning(f"Bu dosya daha önce işlenmiş: {file_hash}")
                        return True
                except Exception:
                    continue
        return False
    except Exception as e:
        logger.error(f"JSONL okuma hatası: {e}")
        return False

def get_existing_chunk_hashes(collection_name):
    doc_path = os.path.join('db/collections', collection_name, 'documents.jsonl')
    hashes = set()
    if os.path.exists(doc_path):
        with open(doc_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if 'text' in obj:
                        hashes.add(calculate_chunk_hash(obj['text']))
                    if 'source' in obj and obj['source_type'] in ["url_web_combined", "web_search"]:
                        hashes.add(calculate_chunk_hash(obj.get('source','')))
                except Exception:
                    continue
    return hashes

def save_chunks_to_jsonl(chunks, jsonl_path):
    """
    Chunk listesini belirtilen JSONL dosyasına ekler.
    """
    import os, json
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        for chunk in chunks:
            # Ensure all text fields are properly encoded as UTF-8
            if 'text' in chunk and isinstance(chunk['text'], str):
                chunk['text'] = chunk['text'].encode('utf-8', errors='replace').decode('utf-8')
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    logger.info(f"{len(chunks)} chunk kaydedildi: {jsonl_path}")

def load_chunks_from_jsonl(jsonl_path):
    """
    JSONL dosyasından chunk listesini yükler.
    """
    import os, json
    chunks = []
    if not os.path.exists(jsonl_path):
        logger.warning(f"Belirtilen chunk dosyası bulunamadı: {jsonl_path}")
        return chunks
    try:
        with open(jsonl_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                    # Ensure text is properly decoded
                    if 'text' in chunk and isinstance(chunk['text'], str):
                        chunk['text'] = chunk['text'].encode('utf-8', errors='replace').decode('utf-8')
                    chunks.append(chunk)
                except Exception as e:
                    logger.error(f"Chunk yükleme hatası: {e}")
                    continue
        logger.info(f"{len(chunks)} chunk yüklendi: {jsonl_path}")
        return chunks
    except Exception as e:
        logger.error(f"Chunks dosyası yükleme hatası: {e}")
        return [] 