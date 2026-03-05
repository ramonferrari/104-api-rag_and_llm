#
# call_llm.py
# 

import re
import logging
from .call_llm import call_llm

logger = logging.getLogger(__name__)

def limpar_markdown_fences(texto: str) -> str:
    if not texto: return ""
    texto = re.sub(r"^```[a-zA-Z]*\n", "", texto.strip(), flags=re.IGNORECASE)
    texto = re.sub(r"\n```$", "", texto.strip())
    return texto.strip()

def orquestrar_chamada(system_prompt: str, user_prompt: str, historico: list, provider='azure', temperature=0.1):
    """
    Orquestrador com MEMÓRIA. 
    'historico' deve ser a lista st.session_state.messages.
    """
    max_out = 32768
    
    # Monta as mensagens: System + Histórico + Pergunta Atual
    messages = [{"role": "system", "content": system_prompt}]
    
    # Adiciona histórico (apenas role e content)
    for m in historico:
        messages.append({"role": m["role"], "content": m["content"]})
    
    # Adiciona a pergunta atual
    messages.append({"role": "user", "content": user_prompt})

    resposta_bruta, métricas = call_llm(
        messages, 
        temperature=temperature, 
        max_tokens=max_out, 
        provider=provider
    )
    
    return limpar_markdown_fences(resposta_bruta), métricas