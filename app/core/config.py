import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

def get_env(key, default=None):
    return os.getenv(key, default)

# Simple settings container
class Settings:
    def __init__(self):
        self.embedding_model = get_env("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.generation_model = get_env("GENERATION_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        self.retrieval_k = int(get_env("RETRIEVAL_K", 5))
        self.chunk_size = int(get_env("CHUNK_SIZE", 500))
        self.chunk_overlap = int(get_env("CHUNK_OVERLAP", 100))
        self.temperature = float(get_env("TEMPERATURE", 0.7))
        self.max_tokens = int(get_env("MAX_TOKENS", 600))
        self.hf_token = get_env("HUGGINGFACEHUB_API_TOKEN")
        # system prompt fallback
        self.system_prompt = (
            """Eres un asesor experto en inmobiliaria. A partir del contexto dado, haz un diagnóstico breve, da recomendaciones accionables, proporciona ejemplos de copy si aplica y aclara supuestos. No inventes información que no esté en el contexto."""
        )

settings = Settings()
