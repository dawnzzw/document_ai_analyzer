import os
import shutil
from pathlib import Path
from typing import List
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import config


def ensure_directories():
    """确保必要的目录存在"""
    Path(config.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.UPLOADED_DOCS_PATH).mkdir(parents=True, exist_ok=True)


def load_document(file_path: str):
    """加载不同格式的文档"""
    file_ext = Path(file_path).suffix.lower()

    if file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_ext in [".txt", ".md"]:
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")

    return loader.load()


def split_documents(documents, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
    """分割文档"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


def create_vector_store(documents, store_name: str):
    """创建向量库"""
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    store_path = os.path.join(config.VECTOR_STORE_PATH, store_name)
    vectorstore.save_local(store_path)
    return store_path


def load_vector_store(store_name: str):
    """加载向量库"""
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    store_path = os.path.join(config.VECTOR_STORE_PATH, store_name)
    return FAISS.load_local(store_path, embeddings)


def save_uploaded_file(uploaded_file, filename: str):
    """保存上传的文件"""
    file_path = os.path.join(config.UPLOADED_DOCS_PATH, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def get_stored_vector_stores() -> List[str]:
    """获取已存储的向量库列表"""
    if not os.path.exists(config.VECTOR_STORE_PATH):
        return []
    return [d for d in os.listdir(config.VECTOR_STORE_PATH)
            if os.path.isdir(os.path.join(config.VECTOR_STORE_PATH, d))]


def delete_vector_store(store_name: str):
    """删除向量库"""
    store_path = os.path.join(config.VECTOR_STORE_PATH, store_name)
    if os.path.exists(store_path):
        shutil.rmtree(store_path)
        return True
    return False


def get_file_size_mb(file_size_bytes: int) -> float:
    """将字节转换为 MB"""
    return file_size_bytes / (1024 * 1024)