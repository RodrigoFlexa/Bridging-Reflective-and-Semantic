"""
Factory de modelos: retorna a instância correta de LLM dado nome e provedor.
"""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

# Limite de tokens gerados por resposta no Ollama.
# Evita que o modelo entre em loop de raciocínio e trave por minutos.
_OLLAMA_MAX_TOKENS = 1024

# Timeout HTTP em segundos para chamadas ao Ollama.
# Se o servidor travar (swap, sobrecarga de GPU), a chamada falha em vez de bloquear.
_OLLAMA_HTTP_TIMEOUT = 120


def load_model(model_name: str, provider: str = "ollama", temperature: float = 0.0):
    """
    Retorna um LLM LangChain.

    Parâmetros
    ----------
    model_name : str
        Nome do modelo (ex: 'llama3.1:latest', 'gpt-4o-mini', 'phi:latest').
    provider : str
        'ollama' ou 'openai'.
    temperature : float
        Temperatura de geração (0 = determinístico).
    """
    provider = provider.lower()
    if provider == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=_OLLAMA_MAX_TOKENS,
            # Passa timeout ao cliente HTTP subjacente (httpx)
            client_kwargs={"timeout": _OLLAMA_HTTP_TIMEOUT},
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=_OLLAMA_MAX_TOKENS,
        )
    else:
        raise ValueError(
            f"Provedor desconhecido: '{provider}'. Use 'ollama' ou 'openai'."
        )
