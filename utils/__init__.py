# utils paketi
# Bu dosya, utils klasörünün bir Python paketi olarak tanınmasını sağlar 

def excelden_metni_al(dosya_yolu):
    from langchain.document_loaders import UnstructuredExcelLoader
    loader = UnstructuredExcelLoader(dosya_yolu)
    docs = loader.load()
    return [doc.page_content for doc in docs] 