# test_3_health.py

import os
import socket
import requests
import torch
import sys
from configparser import ConfigParser

# Cores para o terminal
class Cores:
    OK = '\033[92m'
    AVISO = '\033[93m'
    FALHA = '\033[91m'
    FIM = '\033[0m'
    NEGRITO = '\033[1m'

def check(nome, condicao, dica=""):
    status = f"{Cores.OK}🟢 PASSOU{Cores.FIM}" if condicao else f"{Cores.FALHA}🔴 FALHOU{Cores.FIM}"
    print(f"{nome:<40} {status}")
    if not condicao and dica:
        print(f"   └─ {Cores.AVISO}Dica: {dica}{Cores.FIM}")
    return condicao

def run_health_check():
    print(f"\n{Cores.NEGRITO}🔍 ExtrAI - System Health Check{Cores.FIM}")
    print("="*50)

    # 1. Arquivos Essenciais
    print(f"\n{Cores.NEGRITO}[1] Integridade de Arquivos{Cores.FIM}")
    check("Arquivo config.ini", os.path.exists("config.ini"), "Crie a partir do config.ini.example")
    check("Pasta vector_db/", os.path.exists("vector_db") and len(os.listdir("vector_db")) > 0, "Rode o ingest.py primeiro")
    
    # 2. Configurações de Hardware
    print(f"\n{Cores.NEGRITO}[2] Recursos de Hardware{Cores.FIM}")
    mps_on = torch.backends.mps.is_available()
    cuda_on = torch.cuda.is_available()
    check("Aceleração Apple Silicon (MPS)", mps_on, "Necessário para rodar rápido no seu M3 Pro")
    check("Aceleração NVIDIA (CUDA)", cuda_on, "Necessário se estiver no CDI/HPC")
    
    # 3. Conectividade e Redes
    print(f"\n{Cores.NEGRITO}[3] Conectividade de Rede{Cores.FIM}")
    
    # Teste LM Studio
    is_local = False
    try:
        with socket.create_connection(("localhost", 1234), timeout=1):
            is_local = True
    except: pass
    check("Servidor Local (LM Studio)", is_local, "Inicie o servidor no LM Studio se estiver em casa")

    # Teste Internet / Proxy
    config = ConfigParser()
    config.read('config.ini')
    proxy = config['PROXY'].get('host', '') if 'PROXY' in config else ''
    
    try:
        # Tenta bater no Google ou no Gateway da Petrobras
        requests.get("https://www.google.com", timeout=2)
        net_ok = True
    except:
        net_ok = False
    check("Acesso à Internet/Gateway", net_ok, f"Verifique o Proxy ({proxy}) ou a VPN")

    # 4. Certificados (Ambiente Petrobras)
    if not is_local:
        print(f"\n{Cores.NEGRITO}[4] Segurança Corporativa{Cores.FIM}")
        cert = config['OPENAI'].get('CERTIFICATE_PATH', '') if 'OPENAI' in config else ''
        check("Certificado SSL Petrobras", os.path.exists(cert) if cert else False, "Verifique o caminho do .pem no config.ini")

    print("\n" + "="*50)
    print(f"{Cores.NEGRITO}Check concluído!{Cores.FIM}\n")

if __name__ == "__main__":
    run_health_check()