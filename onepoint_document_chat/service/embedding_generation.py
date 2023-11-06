import shutil
from typing import List, TypeVar

from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document

from onepoint_document_chat.config import cfg
from onepoint_document_chat.log_init import logger
from onepoint_document_chat.service.text_enhancement import enhance_documents
from onepoint_document_chat.service.text_extraction import load_pdfs, combine_documents

VST = TypeVar("VST", bound="VectorStore")


def generate_faiss_embeddings(documents: List[Document]) -> VST:
    """
    Receives a list of documents and generates the embeddings via OpenAI API.

    Parameters:
    documents (List[Document]): The document list with one page per document.

    Returns:
    VST: Recturs a reference to the vector store.
    """
    try:
        docsearch = FAISS.from_documents(documents, cfg.embeddings)
        docsearch.save_local(cfg.embeddings_folder_faiss)
        logger.info("Vector database persisted")
    except Exception as e:
        logger.exception(f"Failed to process documents")
        if "docsearch" in vars() or "docsearch" in globals():
            docsearch.persist()
        return None
    return docsearch


def generate_faiss_enhanced_embeddings(documents: List[Document]) -> VST:
    documents = combine_documents(documents)
    enhanced = enhance_documents(documents)
    return generate_faiss_embeddings(enhanced)


def add_embeddings(documents: List[Document], vst: FAISS):
    vst.add_documents(documents)
    if cfg.embeddings_folder_faiss.exists():
        shutil.rmtree(cfg.embeddings_folder_faiss)
    cfg.embeddings_folder_faiss.mkdir(parents=True, exist_ok=True)
    vst.save_local(cfg.embeddings_folder_faiss)


if __name__ == "__main__":
    documents: List[Document] = load_pdfs(cfg.data_folder)
    vst = generate_faiss_enhanced_embeddings(documents)
    assert vst is not None
