"""
Factory de modelos: retorna a instância correta de LLM dado nome e provedor.
"""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


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
        return ChatOllama(model=model_name, temperature=temperature)
    elif provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(
            f"Provedor desconhecido: '{provider}'. Use 'ollama' ou 'openai'."
        )
