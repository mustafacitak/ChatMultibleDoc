import re
import yaml
from loguru import logger
from typing import List

def get_chunking_config(config_path='config/chunking_config.yaml'):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        chunk_size = config.get('chunking', {}).get('chunk_size', 5)
        overlap = config.get('chunking', {}).get('overlap', 0)
        return int(chunk_size), int(overlap)
    except Exception:
        return 5, 0

def dynamic_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    # Ensure text is properly encoded
    sanitized_text = text.encode('utf-8', errors='replace').decode('utf-8') if isinstance(text, str) else str(text)
    words = sanitized_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            # Ensure chunk is properly encoded
            chunk = chunk.encode('utf-8', errors='replace').decode('utf-8')
            chunks.append(chunk)
    logger.info(f"{len(chunks)} adet chunk üretildi.")
    return chunks

def sentence_chunking(text: str, chunk_size: int = 5) -> List[str]:
    # Ensure text is properly encoded
    sanitized_text = text.encode('utf-8', errors='replace').decode('utf-8') if isinstance(text, str) else str(text)
    sentences = re.split(r'(?<=[.!?]) +', sanitized_text)
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i+chunk_size])
        if chunk:
            # Ensure chunk is properly encoded
            chunk = chunk.encode('utf-8', errors='replace').decode('utf-8')
            chunks.append(chunk)
    logger.info(f"{len(chunks)} adet cümle chunk üretildi.")
    return chunks

def dynamic_chunking_advanced(text, chunk_size=None, overlap=None):
    """
    Paragrafları, başlıkları ve cümleleri dikkate alarak dinamik chunking yapar.
    Her chunk meta veri içerir: başlık, paragraf no, chunk no.
    """
    if chunk_size is None or overlap is None:
        chunk_size, overlap = get_chunking_config()
    # Paragraflara böl
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    chunk_id = 0
    for para_idx, para in enumerate(paragraphs):
        # Başlık tespiti (ör: satırın tamamı büyük harf veya : ile bitiyorsa)
        lines = para.split('\n')
        title = None
        if len(lines) > 0 and (lines[0].isupper() or lines[0].strip().endswith(':')):
            title = lines[0].strip()
            para_body = '\n'.join(lines[1:])
        else:
            para_body = para
        # Cümlelere böl
        sentences = re.split(r'(?<=[.!?]) +', para_body)
        # Dinamik olarak cümleleri birleştirerek chunk oluştur
        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i:i+chunk_size]
            chunk_text = ' '.join(chunk_sentences).strip()
            if chunk_text:
                meta = {
                    'chunk_id': chunk_id,
                    'paragraph_id': para_idx,
                    'title': title,
                }
                chunks.append({'text': chunk_text, 'meta': meta})
                chunk_id += 1
            i += chunk_size - overlap
    return chunks 