from typing import List, TypeVar
from pathlib import Path
import shutil
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from onepoint_document_chat.service.embedding_generation import (
    generate_faiss_enhanced_embeddings,
)

from onepoint_document_chat.config import cfg
from onepoint_document_chat.log_init import logger
from onepoint_document_chat.service.text_extraction import load_pdfs
from onepoint_document_chat.service.vector_search_common import similarity_search


def init_vector_search(delete: bool = False) -> FAISS:
    embedding_dir = cfg.embeddings_folder_faiss.as_posix()
    embedding_dir_path = Path(embedding_dir)
    if delete:
        shutil.rmtree(embedding_dir_path)
    # Check if directory exists and has something inside
    if embedding_dir_path.exists() and len(list(embedding_dir_path.glob("*"))) > 0:
        logger.info(f"reading from existing directory")
        return FAISS.load_local(embedding_dir, cfg.embeddings)
    else:
        logger.info(f"creating new faiss index")
        documents: List[Document] = load_pdfs(cfg.data_folder)
        return generate_faiss_enhanced_embeddings(documents)


if __name__ == "__main__":
    vst = init_vector_search(False)
    documents = similarity_search(
        vst, "Which are Onepoint's projects related to the travel industry?"
    )
    logger.info(documents)
