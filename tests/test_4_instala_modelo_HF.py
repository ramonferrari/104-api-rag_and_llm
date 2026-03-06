import os
import ssl
import getpass
from configparser import ConfigParser
from huggingface_hub import snapshot_download

def download_bge_model():
    target_dir = "./models/bge-m3"
    config = ConfigParser()
    config.read('config.ini')
    
    # Pega o host e limpa espaços e o protocolo http:// se existir
    proxy_host = config['PROXY'].get('host', '').strip().replace('http://', '').replace('https://', '')

    # SE O HOST ESTIVER VAZIO (Cenário Casa)
    if not proxy_host:
        print("🏠 [HOME] Host vazio no config.ini. Usando conexão direta.")
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)
    
    # SE O HOST TIVER ALGO (Cenário Trabalho)
    else:
        print(f"🏢 [PETROBRAS] Detectado host: {proxy_host}")
        user = input("Usuário (chave): ")
        password = getpass.getpass("Senha da Rede: ")
        
        # Monta a URL. Se o proxy_host já tiver a porta (ex: :804), o f-string resolve.
        proxy_url = f"http://{user}:{password}@{proxy_host}"
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url

    # Configurações padrão para o download
    ssl._create_default_https_context = ssl._create_unverified_context
    print(f"🚀 Baixando 2.2GB para {target_dir}...")

    try:
        snapshot_download(
            repo_id="BAAI/bge-m3",
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        print("\n✅ Sucesso! O M3 Pro já pode começar a trabalhar.")
    except Exception as e:
        print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    download_bge_model()