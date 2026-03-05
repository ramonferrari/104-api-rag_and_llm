import logging
import json
import tiktoken
from configparser import ConfigParser, ExtendedInterpolation
from openai import OpenAI
from .get_azure_openai_client import get_azure_openai_client
from .get_aws_bedrock_client import get_aws_bedrock_client

CONFIG_FILE = "config.ini"
logger = logging.getLogger(__name__)

def calcular_tokens_locais(messages):
    """Calcula tokens localmente usando tiktoken (fallback)."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base") # Padrão GPT-4/Claude
        texto_total = "".join([m['content'] for m in messages if isinstance(m['content'], str)])
        return len(encoding.encode(texto_total))
    except:
        return 0

def call_llm(messages, temperature=0.3, max_tokens=30000, model_name=None, provider='azure'):
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(CONFIG_FILE, 'UTF-8')
    
    metrics = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # --- ROTA 1: AWS BEDROCK (STREAMING) ---
    if provider == 'aws':
        client = get_aws_bedrock_client()
        
        # VERIFICAÇÃO DE SEGURANÇA PARA O MAC EM CASA
        if client is None:
            logger.warning("⚠️ AWS Bedrock indisponível (iaaws_lib não encontrada).")
            # Se você estiver no Streamlit, pode querer um fallback automático para 'local'
            # return call_llm(messages, temperature, max_tokens, model_name, provider='local')
            raise RuntimeError("Biblioteca iaaws_lib necessária para AWS não encontrada neste ambiente.")

        if model_name is None: 
            model_name = config['AWS']['BEDROCK_CLAUDE_MODEL_ID']
        
        system_content = next((m['content'] for m in messages if m['role'] == 'system'), "")
        user_messages = [{"role": m['role'], "content": [{"type": "text", "text": m['content']}]} 
                         for m in messages if m['role'] != 'system']

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_content,
            "messages": user_messages
        }

        try:
            response = client.invoke_model_with_response_stream(
                body=json.dumps(payload), modelId=model_name
            )
            full_text = ""
            for event in response.get('body'):
                chunk = event.get('chunk')
                if chunk:
                    chunk_obj = json.loads(chunk.get('bytes').decode())
                    if 'delta' in chunk_obj and 'text' in chunk_obj['delta']:
                        full_text += chunk_obj['delta']['text']
            
            metrics["prompt_tokens"] = calcular_tokens_locais(messages)
            metrics["completion_tokens"] = calcular_tokens_locais([{"content": full_text}])
            metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
            
            return full_text, metrics
        except Exception as e:
            logger.error(f"Erro AWS: {e}")
            raise e

    # --- ROTA 2: LOCAL (LM STUDIO) ---
    elif provider == 'local':
        try:
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            response = client.chat.completions.create(
                model="local-model", messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            usage = response.usage
            metrics["prompt_tokens"] = getattr(usage, 'prompt_tokens', calcular_tokens_locais(messages))
            metrics["completion_tokens"] = getattr(usage, 'completion_tokens', calcular_tokens_locais([{"content": response.choices[0].message.content}]))
            metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
            
            return response.choices[0].message.content, metrics
        except Exception as e:
            logger.error(f"Erro Local: {e}")
            raise e

    # --- ROTA 3: AZURE OPENAI ---
    else:
        if model_name is None: 
            model_name = config['OPENAI']['CHATGPT_MODEL']
        
        # Note: get_azure_openai_client também deve ser blindado se usar libs internas
        client = get_azure_openai_client()
        
        if client is None:
             raise RuntimeError("Cliente Azure OpenAI não disponível neste ambiente.")

        try:
            response = client.chat.completions.create(
                model=model_name, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            usage = response.usage
            metrics = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
            return response.choices[0].message.content, metrics
        except Exception as e:
            logger.error(f"Erro Azure: {e}")
            raise e