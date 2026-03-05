# 1_🕵️_RAG_System.py
import streamlit as st
import os
import torch
import socket
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Utilitários
from utils.llm_orchestrator import orquestrar_chamada
from global_utils import apply_global_styles

# --- 1. FUNÇÕES DE APOIO ---

def testar_conexao(host, porta):
    try:
        with socket.create_connection((host, porta), timeout=0.5):
            return True
    except:
        return False

@st.cache_resource
def load_rag_assets():
    try:
        # 1. Caminho relativo (funciona em ambos)
        model_path = "./models/bge-m3"
        
        # 2. Offline total para não bater em firewall nenhum
        os.environ['TRANSFORMERS_OFFLINE'] = "1"
        os.environ['HF_HUB_OFFLINE'] = "1"

        # 3. Inteligência de Hardware
        if torch.backends.mps.is_available():
            device = 'mps'  # Seu MacBook M3 Pro entra aqui
        elif torch.cuda.is_available():
            device = 'cuda' # Caso o Windows tenha NVIDIA
        else:
            device = 'cpu'  # Fallback padrão
            
        print(f"🧠 Inicializando ExtrAI via {device}...")

        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': device, 'local_files_only': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        return Chroma(persist_directory="./vector_db", embedding_function=embeddings)
    except Exception as e:
        st.error(f"Erro ao despertar o cérebro: {e}")
        return None


# --- 2. INTERFACE PRINCIPAL ---

def main():
    st.set_page_config(page_title="ExtrAI Research", page_icon="🕵️", layout="wide")
    apply_global_styles()
    st.title("🕵️ ExtrAI - Research Intelligence")

    is_local_online = testar_conexao("localhost", 1234)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configurações")
        mapa_modelos = {
            f"{'🟢' if is_local_online else '🔴'} LM Studio (Local)": "local",
            "🟦 GPT-4o (Azure)": "azure", 
            "🟧 Claude 3.5 (AWS)": "aws"
        }
        label_selecionada = st.radio("IA Engine:", list(mapa_modelos.keys()), 
                                    index=0 if is_local_online else 1)
        provider = mapa_modelos[label_selecionada]
        
        st.divider()
        top_k = st.slider("Contexto (Top-K)", 5, 20, 10)
        temperature = st.slider("Temperatura", 0.0, 1.0, 0.2, step=0.1)
        
        if st.button("🗑️ Resetar Pesquisa"):
            st.session_state.messages = []
            st.rerun()

    # --- CARREGAMENTO DO BANCO (LÓGICA BLINDADA) ---
    # --- D. Carregamento do Banco de Dados ---
    db = load_rag_assets()
    
    if db is None:
        st.error("❌ O Banco de Vetores não pôde ser inicializado.")
        st.info("Verifique o erro detalhado acima.")
        if st.button("♻️ Tentar Novamente"):
            st.cache_resource.clear()
            st.rerun()
        st.stop() # PARA TUDO AQUI SE DER ERRO
    
    # SE CHEGOU AQUI, O DB É VÁLIDO. TUDO DEVE ESTAR DENTRO DO TRY OU APÓS O STOP
    try:
        data = db.get()
        num_docs = len(data['ids']) if data and 'ids' in data else 0
        st.caption(f"📚 Base: {num_docs} seções | 🔌 LM Studio: {'ON' if is_local_online else 'OFF'}")
    except Exception as e:
        st.warning(f"⚠️ Banco conectado, mas erro ao ler seções: {e}")

    # --- INTERFACE DE CHAT ---
    st.markdown("""<style>
        .azure-theme { background-color: rgba(16, 185, 129, 0.1); border-left: 5px solid #10B981; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
        .aws-theme { background-color: rgba(249, 115, 22, 0.1); border-left: 5px solid #F97316; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
        .local-theme { background-color: rgba(139, 92, 246, 0.1); border-left: 5px solid #8B5CF6; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    </style>""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                theme = f"{msg.get('provider', 'azure')}-theme"
                st.markdown(f"<div class='{theme}'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

    # --- FLUXO DE PERGUNTA ---
    if query := st.chat_input("Pergunte sobre sua base..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"): st.markdown(query)

        with st.status(f"🚀 {provider.upper()} analisando literatura...", expanded=True) as status:
            try:
                results = db.similarity_search_with_relevance_scores(query, k=top_k)
                if not results:
                    st.warning("Nenhum trecho relevante.")
                    st.stop()

                contexto = "".join([f"\n[Fonte: {d.metadata.get('source')}]: {d.page_content}\n" for d, s in results])
                fontes = [f"**{d.metadata.get('source')}**" for d, s in results]

                system_prompt = """Você é um assistente de pesquisa acadêmico rigoroso e de alto nível. Use o contexto advindo de literatura acadêmica para responder. REGRAS DE FORMATAÇÃO:
                1. NUNCA use nomes completos (ex: Moshe Vardi). Use apenas o SOBRENOME seguido do ano entre parênteses.
                2. Formato obrigatório: Sobrenome (Ano). Exemplo: Vardi (2020) ou Shalf (2023).
                3. Se houver dois autores, use: Sobrenome1 e Sobrenome2 (Ano).
                4. Se houver mais de dois, use: Sobrenome et al. (Ano).
                5. Se o ano não estiver disponível no contexto, use (n.d.).
                6. Mantenha o tom formal e impessoal."""
                user_prompt = f"CONTEXTO:\n{contexto}\n\nPERGUNTA: {query}"
                
                resposta, métricas = orquestrar_chamada(system_prompt, user_prompt, 
                                                       st.session_state.messages[:-1], 
                                                       provider, temperature)

                st.session_state.messages.append({
                    "role": "assistant", "content": resposta,
                    "sources": "\n\n".join(list(set(fontes))),
                    "provider": provider, "metrics": métricas
                })
                status.update(label="✅ Concluído!", state="complete")
                st.rerun()
            except Exception as e:
                st.error(f"Erro no processamento: {e}")

if __name__ == "__main__":
    main()