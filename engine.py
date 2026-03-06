#
# engine.py 
# 
import os
import torch
import socket
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils.llm_orchestrator import orquestrar_chamada

class RAGEngine:
    def __init__(self):
        try:
            self.db = self._load_rag_assets()
            print("✅ RAG Assets carregados com sucesso.")
        except Exception as e:
            print(f"❌ Erro crítico ao carregar banco: {e}")
            self.db = None
    
    def _load_rag_assets(self):
        model_path = "./models/bge-m3"
        os.environ['TRANSFORMERS_OFFLINE'] = "1"
        os.environ['HF_HUB_OFFLINE'] = "1"

        if torch.backends.mps.is_available(): device = 'mps'
        elif torch.cuda.is_available(): device = 'cuda'
        else: device = 'cpu'
            
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': device, 'local_files_only': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        return Chroma(persist_directory="./vector_db", embedding_function=embeddings)

    def check_local_llm(self):
        try:
            with socket.create_connection(("localhost", 1234), timeout=0.5):
                return True
        except:
            return False

    import os

    def query(self, user_query: str, history: list, provider: str, top_k: int, temperature: float):
        # 1. Busca semântica no ChromaDB
        results = self.db.similarity_search_with_relevance_scores(user_query, k=top_k)
        
        if not results:
            return "Desculpe, não encontrei informações relevantes na base técnica.", [], {}

        # 2. Preparação das Fontes Estruturadas (para o Frontend) e Contexto (para a LLM)
        fontes_estruturadas = []
        contexto_parts = []
        
        # Usamos um set para evitar duplicar trechos idênticos se o banco retornar sobreposição
        vistos = set()

        for doc, score in results:
            source_path = doc.metadata.get('source', 'documento_desconhecido.md')
            file_name = os.path.basename(source_path) # Remove o caminho e deixa só o nome.md
            
            # Criamos um ID único para o trecho para evitar redundância visual
            content_hash = hash(doc.page_content[:50])
            
            if content_hash not in vistos:
                fontes_estruturadas.append({
                    "document": file_name,
                    "section": "Markdown Source", # Como não há páginas, indicamos a natureza do arquivo
                    "snippet": doc.page_content[:300].replace("\n", " ") + "..." # Limpa quebras de linha no snippet
                })
                vistos.add(content_hash)
            
            contexto_parts.append(f"\n[Fonte: {file_name}]: {doc.page_content}\n")

        contexto_completo = "".join(contexto_parts)

        # 3. Definição dos Prompts
        system_prompt = """Você é um assistente de pesquisa acadêmico rigoroso e de alto nível. 
        Use o contexto advindo de literatura técnica e científica fornecido para responder. 
        REGRAS DE FORMATAÇÃO:
        1. NUNCA use nomes completos (ex: Moshe Vardi). Use apenas o SOBRENOME seguido do ano entre parênteses.
        2. Formato obrigatório: Sobrenome (Ano). Exemplo: Vardi (2020) ou Shalf (2023).
        3. Se houver dois autores, use: Sobrenome1 e Sobrenome2 (Ano).
        4. Se houver mais de dois, use: Sobrenome et al. (Ano).
        5. Se o ano não estiver disponível no contexto, use (n.d.).
        6. Mantenha o tom formal, impessoal e objetivo."""

        user_prompt_full = f"CONTEXTO EXTRAÍDO DA BASE:\n{contexto_completo}\n\nPERGUNTA DO USUÁRIO: {user_query}"
        
        # 4. Chamada ao Orquestrador (OpenAI/LM Studio/Azure/AWS)
        try:
            resposta, métricas = orquestrar_chamada(
                system_prompt, 
                user_prompt_full, 
                history, 
                provider, 
                temperature
            )
            # Retorna a resposta da IA, a lista de dicionários para o Accordion e as métricas de tokens/tempo
            return resposta, fontes_estruturadas, métricas

        except Exception as e:
            print(f"Erro na orquestração: {e}")
            return f"Erro ao processar a resposta da IA: {str(e)}", fontes_estruturadas, {}