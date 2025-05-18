import requests
import asyncio
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
from loguru import logger
import functools
import aiohttp

# Hata yakalama dekoratörü
def search_error_handler(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Arama hatası - Fonksiyon: {func.__name__}, Hata: {e}")
            return []
    return wrapper

# DuckDuckGo Search
@search_error_handler
async def search_duckduckgo(query, max_results=5):
    async with aiohttp.ClientSession() as session:
        url = f"https://duckduckgo.com/html/?q={query}"
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    results = []
                    for a in soup.select('.result__a')[:max_results]:
                        results.append({'title': a.text, 'url': a['href']})
                    logger.info(f"DuckDuckGo araması başarılı: {len(results)} sonuç bulundu.")
                    return results
                logger.warning(f"DuckDuckGo araması başarısız, durum kodu: {resp.status}")
                return []
        except Exception as e:
            logger.error(f"DuckDuckGo araması sırasında hata: {e}")
            return []

# OpenAlex Search (akademik makale)
@search_error_handler
async def search_openalex(query, max_results=5):
    async with aiohttp.ClientSession() as session:
        url = f"https://api.openalex.org/works?search={query}&per-page={max_results}"
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = []
                    for item in data.get('results', [])[:max_results]:
                        results.append({'title': item.get('title'), 'url': item.get('id')})
                    logger.info(f"OpenAlex araması başarılı: {len(results)} sonuç bulundu.")
                    return results
                logger.warning(f"OpenAlex araması başarısız, durum kodu: {resp.status}")
                return []
        except Exception as e:
            logger.error(f"OpenAlex araması sırasında hata: {e}")
            return []

# arXiv Search (preprint makale)
@search_error_handler
async def search_arxiv(query, max_results=5):
    async with aiohttp.ClientSession() as session:
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    root = ET.fromstring(text)
                    results = []
                    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                        title = entry.find('{http://www.w3.org/2005/Atom}title').text
                        link = entry.find('{http://www.w3.org/2005/Atom}id').text
                        results.append({'title': title, 'url': link})
                    logger.info(f"arXiv araması başarılı: {len(results)} sonuç bulundu.")
                    return results[:max_results]
                logger.warning(f"arXiv araması başarısız, durum kodu: {resp.status}")
                return []
        except Exception as e:
            logger.error(f"arXiv araması sırasında hata: {e}")
            return []

# Typesense Search - Gerçek implementasyon
@search_error_handler
async def search_typesense(query, max_results=5):
    try:
        # Typesense API'ye bağlanma
        import typesense
        client = typesense.Client({
            'api_key': 'xyz',  # Gerçek uygulamada, bu değer config'den gelmeli
            'nodes': [{
                'host': 'localhost',
                'port': '8108',
                'protocol': 'http'
            }],
            'connection_timeout_seconds': 2
        })
        
        # Aramayı gerçekleştir
        search_parameters = {
            'q': query,
            'query_by': 'title,content',
            'per_page': max_results
        }
        
        search_results = client.collections['documents'].documents.search(search_parameters)
        
        results = []
        for hit in search_results['hits']:
            results.append({
                'title': hit['document'].get('title', 'Başlık yok'),
                'url': hit['document'].get('url', '#')
            })
            
        logger.info(f"Typesense araması başarılı: {len(results)} sonuç bulundu.")
        return results
    except Exception as e:
        logger.error(f"Typesense araması sırasında hata: {e}")
        return []

# Meilisearch Search - Gerçek implementasyon
@search_error_handler
async def search_meilisearch(query, max_results=5):
    try:
        # Meilisearch API'ye bağlanma
        from meilisearch import Client
        client = Client('http://localhost:7700', 'masterKey')  # Gerçek değerler config'den gelmeli
        
        # Aramayı gerçekleştir
        search_results = client.index('documents').search(query, {
            'limit': max_results
        })
        
        results = []
        for hit in search_results['hits']:
            results.append({
                'title': hit.get('title', 'Başlık yok'),
                'url': hit.get('url', '#')
            })
            
        logger.info(f"Meilisearch araması başarılı: {len(results)} sonuç bulundu.")
        return results
    except Exception as e:
        logger.error(f"Meilisearch araması sırasında hata: {e}")
        return []

# OpenSearch Search - Gerçek implementasyon
@search_error_handler
async def search_opensearch(query, max_results=5):
    try:
        # OpenSearch API'ye bağlanma
        from opensearchpy import AsyncOpenSearch
        
        client = AsyncOpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200}],
            http_auth=('admin', 'admin'),  # Gerçek değerler config'den gelmeli
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # Aramayı gerçekleştir
        search_body = {
            'query': {
                'multi_match': {
                    'query': query,
                    'fields': ['title', 'content']
                }
            },
            'size': max_results
        }
        
        search_results = await client.search(index='documents', body=search_body)
        
        results = []
        for hit in search_results['hits']['hits']:
            source = hit['_source']
            results.append({
                'title': source.get('title', 'Başlık yok'),
                'url': source.get('url', '#')
            })
            
        logger.info(f"OpenSearch araması başarılı: {len(results)} sonuç bulundu.")
        return results
    except Exception as e:
        logger.error(f"OpenSearch araması sırasında hata: {e}")
        return []

# Ana net_search fonksiyonu (async)
async def net_search_async(query, max_results=5):
    logger.info(f"Web araması başlatıldı: {query}")
    # Tüm aramaları paralel olarak çalıştır
    tasks = [
        search_duckduckgo(query, max_results),
        search_openalex(query, max_results),
        search_arxiv(query, max_results),
        search_typesense(query, max_results),
        search_meilisearch(query, max_results),
        search_opensearch(query, max_results)
    ]
    
    results = []
    completed_tasks = await asyncio.gather(*tasks)
    
    # Tüm sonuçları birleştir
    for task_result in completed_tasks:
        results.extend(task_result)
    
    logger.info(f"Web araması tamamlandı: Toplam {len(results)} sonuç bulundu.")
    return results

# Geriye dönük uyumluluk için senkron wrapper
def net_search(query, max_results=5):
    logger.info(f"Senkron net_search çağrıldı: {query}")
    return asyncio.run(net_search_async(query, max_results)) 