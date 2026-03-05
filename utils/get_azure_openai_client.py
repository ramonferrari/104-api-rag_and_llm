# utils/get_azure_openai_client.py
import os
from configparser import ConfigParser, ExtendedInterpolation
from xmlrpc import client
from httpx import Client
from openai import AzureOpenAI

CONFIG_FILE = "config.ini"

def get_azure_openai_client():
    """
    Cliente genérico do Azure OpenAI.
    Reutilizável para QUALQUER app do hub.
    """
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(CONFIG_FILE, 'UTF-8')
    api_key = config['OPENAI']['OPENAI_API_KEY']
    api_version = config['OPENAI']['OPENAI_API_VERSION']
    base_url = config['OPENAI']['AZURE_OPENAI_BASE_URL']
    http_client = None
    if 'CERTIFICATE_PATH' in config['OPENAI']:
        cert_path = config['OPENAI']['CERTIFICATE_PATH']
        if os.path.exists(cert_path):
            http_client = Client(verify=cert_path)
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=base_url,
        http_client=http_client
    )
    
    return client