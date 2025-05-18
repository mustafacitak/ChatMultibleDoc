from loguru import logger

def classify_query_type(query: str) -> str:
    if any(x in query for x in ["özet", "summary"]):
        qtype = "özet"
    elif any(x in query for x in ["analiz", "analiz et"]):
        qtype = "analiz"
    else:
        qtype = "genel"
    logger.info(f"Sorgu türü: {qtype}")
    return qtype 