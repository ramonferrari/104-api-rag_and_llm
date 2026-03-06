# test_2_query.py

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. MESMAS CONFIGS DE CACHE
os.environ['HF_HOME'] = "/projetos/ubi4/rag_system/.cache/huggingface"

def test_query():
    print("🧠 Carregando modelo e conectando ao banco...")
    
    # IMPORTANTE: Use exatamente as mesmas configs do ingest.py
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    db = Chroma(
        persist_directory="/projetos/ubi4/rag_system/vector_db", 
        embedding_function=embeddings
    )

    # Teste de busca: Pergunta em PORTUGUÊS para documento em INGLÊS
    pergunta = "Qual a relação entre inovação de produto e processo segundo Abernathy?"
    
    print(f"\n🔍 Buscando por: '{pergunta}'")
    
    # k=3 traz os 3 trechos mais relevantes
    docs = db.similarity_search(pergunta, k=3)

    if not docs:
        print("❌ Nenhum trecho encontrado. O banco está vazio?")
        return

    for i, doc in enumerate(docs):
        print(f"\n--- Trecho Recuperado {i+1} ---")
        print(f"📄 Fonte: {doc.metadata.get('source')} | Seção: {doc.metadata.get('section')}")
        print(f"📝 Conteúdo: {doc.page_content[:300]}...")

if __name__ == "__main__":
    test_query()