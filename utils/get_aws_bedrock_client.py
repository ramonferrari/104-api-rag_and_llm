import os
import logging
from configparser import ConfigParser, ExtendedInterpolation
from botocore.config import Config 

# 1. Tenta importar a lib da firma. Se falhar (em casa), define a flag como False.
try:
    from iaaws_lib.aws_helper import AwsHelper
    HAS_IAAWS = True
except ImportError:
    HAS_IAAWS = False

CONFIG_FILE = "config.ini"
logger = logging.getLogger(__name__)

def get_aws_bedrock_client():
    """
    Cliente do AWS Bedrock Runtime usando a lib interna 'iaaws-lib'.
    Preserva a lógica corporativa mas permite execução local (Home).
    """
    
    # 2. Se não tem a lib (Cenário Casa/Mac), avisa e sai graciosamente
    if not HAS_IAAWS:
        logger.warning("🏠 [HOME MODE] iaaws-lib não encontrada. Bedrock desativado.")
        return None

    # 3. LÓGICA ORIGINAL DA FIRMA (100% Preservada)
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(CONFIG_FILE, 'UTF-8')

    try:
        # Configurações de Ambiente e Certificado
        env = 'TST' 
        cert_path = config['PATHS'].get('CERTIFICATE_PATH', None)
        
        # Recuperação da Chave de API
        if 'IAAWS_API_KEY' in config['AWS']:
            api_key = config['AWS']['IAAWS_API_KEY']
        else:
            raise ValueError("Chave de API da iaaws-lib não encontrada no config.ini")

        # CONFIGURAÇÃO DE TIMEOUT (Seu ajuste para evitar o Read Timeout)
        my_boto_config = Config(
            read_timeout=1800, 
            connect_timeout=60,
            retries={'max_attempts': 5, 'mode': 'standard'}
        )

        logger.info(f"🔐 [PETROBRAS] Inicializando AwsHelper (Env: {env})")
        aws_helper = AwsHelper(env, cert_path)

        # Cria o cliente Bedrock com sua config de timeout
        client = aws_helper.create_iaaws_client('bedrock-runtime', api_key)
        
        return client

    except Exception as e:
        logger.error(f"❌ Erro ao criar cliente via iaaws-lib: {e}")
        # Aqui você decide se quer dar 'raise' ou retornar None. 
        # Para o RAG não quebrar o Streamlit, retornar None é mais seguro.
        return None