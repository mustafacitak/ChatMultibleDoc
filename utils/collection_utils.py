import os
import json
import re
from loguru import logger

def normalize_collection_name(name: str) -> str:
    name = name.strip().replace(" ", "_").lower()
    name = re.sub(r'[^a-z0-9_-]', '', name)
    return name or "koleksiyon"

def get_existing_collections(base_path: str = "db/collections") -> list:
    if not os.path.exists(base_path):
        return []
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def create_or_update_collection(docs: list, collection_name: str, source_name: str = None):
    base_path = "db/collections"
    col_path = os.path.join(base_path, collection_name)
    os.makedirs(col_path, exist_ok=True)
    doc_path = os.path.join(col_path, "documents.jsonl")
    with open(doc_path, 'a', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    logger.info(f"{len(docs)} belge {collection_name} koleksiyonuna eklendi.")
    if source_name:
        with open(os.path.join(col_path, "sources.txt"), 'a', encoding='utf-8') as f:
            f.write(source_name + '\n') 