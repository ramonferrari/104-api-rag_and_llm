#
# main.py
#
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from engine import RAGEngine

app = FastAPI(title="ExtrAI API")

# Habilitar CORS para o seu frontend TypeScript (ex: localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Em produção, especifique a URL do seu front
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o motor uma única vez (Singleton)
rag = RAGEngine()

# --- MODELOS DE DADOS (Pydantic) ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[Message] = []
    provider: str = "azure"
    top_k: int = 10
    temperature: float = 0.2

# --- ENDPOINTS ---

@app.get("/status")
def get_status():
    """Retorna se o LM Studio está online e info do banco"""
    return {
        "local_llm_online": rag.check_local_llm(),
        "database_ready": rag.db is not None
    }

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Converte histórico Pydantic para lista de dicts (esperado pelo orquestrador)
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
        
        resposta, fontes, métricas = rag.query(
            req.query, history_dicts, req.provider, req.top_k, req.temperature
        )
        
        return {
            "answer": resposta,
            "sources": fontes,
            "metrics": métricas,
            "provider": req.provider
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)