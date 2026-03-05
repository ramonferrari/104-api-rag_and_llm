import os
import gc
import torch
import urllib.parse
from getpass import getpass
from pathlib import Path
from configparser import ConfigParser

# 1. CARREGAR CONFIGURAÇÕES
config = ConfigParser()
config.read('config.ini', encoding='utf-8')

# Injetar variáveis de ambiente do config
os.environ['PYTORCH_ALLOC_CONF'] = config['ENVIRONMENT']['pytorch_alloc_conf']
os.environ['TRANSFORMERS_OFFLINE'] = config['ENVIRONMENT']['transformers_offline']

# 2. INPUT DE CREDENCIAIS (INTERATIVO)
print("🔐 Acesso à rede da empresa")
# Você pode deixar o CHAVE no config ou pedir no input
CHAVE = getpass("User (ex: JasonMomoa33): ") 
SENHA = getpass(f"Senha para {CHAVE}: ")
senha_encoded = urllib.parse.quote(SENHA)

# Monta o Proxy dinamicamente
proxy_host = config['PROXY']['host']
os.environ['HTTP_PROXY'] = f"http://{CHAVE}:{senha_encoded}@{proxy_host}"
os.environ['HTTPS_PROXY'] = os.environ['HTTP_PROXY']

# Define o Cache relativo à raiz do projeto
BASE_PATH = Path(__file__).parent.resolve()
os.environ['HF_HOME'] = str(BASE_PATH / config['PATHS']['cache_dir'])

# Importações tardias (após configurar proxy/cache)
from src.parser import AcademicParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    # Caminhos relativos
    data_path = BASE_PATH / config['PATHS']['data_dir']
    db_path = BASE_PATH / config['PATHS']['vector_db_dir']

    print(f"\n🔍 Lendo arquivos em: {data_path}")
    parser = AcademicParser(data_path)
    documents = parser.process_all()
    print(f"✅ {len(documents)} seções carregadas.")

    print(f"🧠 Carregando {config['MODELS']['embedding_model']} na GPU...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config['MODELS']['embedding_model'],
        model_kwargs={'device': config['MODELS']['device']},
        encode_kwargs={'normalize_embeddings': True}
    )

    torch.cuda.empty_cache()
    gc.collect()

    print(f"📦 Indexando no ChromaDB em: {db_path}")
    batch_size = 10 
    
    # Primeiro lote
    vector_db = Chroma.from_documents(
        documents=documents[:batch_size],
        embedding=embeddings,
        persist_directory=str(db_path)
    )

    # Restante com limpeza de memória
    for i in range(batch_size, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        print(f"⏳ Lote: {i} até {min(i + batch_size, len(documents))} / {len(documents)}")
        
        vector_db.add_documents(batch)
        torch.cuda.empty_cache()
        gc.collect()

    print("\n🎉 SUCESSO! Banco de dados estruturado e pronto para Git.")

if __name__ == "__main__":
    main()