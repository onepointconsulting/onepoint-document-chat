from typing import List, TypeVar

import tiktoken
from langchain.schema import Document
from onepoint_document_chat.config import cfg
from onepoint_document_chat.log_init import logger


VST = TypeVar("VST", bound="VectorStore")


def join_pages(doc_list: List[Document]) -> str:
    return "\n\n".join([p.page_content for p in doc_list])


def num_tokens_from_string(string: str) -> int:
    """
    Returns the number of tokens in a text string.

    Parameters:
    string (str): The string for which the tiktokens are to be counted.

    Returns:
    int: Recturs the number of tokens generated using tiktoken.
    """
    encoding = tiktoken.encoding_for_model(cfg.model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def similarity_search(
    docsearch: VST, input: str, how_many=cfg.search_results_how_many
) -> List[Document]:
    """
    Performs multiple searches until it reaches the maximum amount of tokens below a specified threshold.
    When the threshold of tokens is reached it stops and returns the search results.

    Parameters:
    docsearch VST: The object used to access the vector database.
    input str: The input of the search.
    how_many int: The initial number of results to be retrieved.

    Returns:
    str: The maximum amount of text with the number of tokens below the threshold specified in the configuration.
    """
    token_count = 0
    previous_res = []
    attempts = 0
    max_attempts = cfg.search_results_extra_attempts
    while attempts < max_attempts:
        doc_list = docsearch.similarity_search(input, k=how_many + attempts)
        logger.info("Similarity search results: %s", len(doc_list))
        joined = join_pages(doc_list)
        token_count = num_tokens_from_string(joined)
        logger.info("Token count: %d", token_count)
        attempts += 1
        if token_count > cfg.context_token_limit:
            return previous_res
        previous_res = doc_list
    return previous_res
