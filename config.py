import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# 应用配置
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# 文本分割配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 向量库配置
VECTOR_STORE_PATH = "data/vector_stores"
UPLOADED_DOCS_PATH = "data/uploaded_docs"

# RAG 配置
TOP_K_RETRIEVAL = 3
TEMPERATURE = 0.7
MAX_TOKENS = 2000