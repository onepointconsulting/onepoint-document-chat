from typing import Dict, Any, Union

from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.chains.openai_functions.base import convert_to_openai_function

from onepoint_document_chat.service.vector_search import (
    init_vector_search,
    similarity_search,
)
from onepoint_document_chat.service.text_extraction import FILE_NAME, PAGE
from onepoint_document_chat.toml_support import prompts_toml
from onepoint_document_chat.config import cfg


SUMMARIES = "summaries"
QUESTION = "question"
HISTORY = "history"


def convert_document_to_text(doc: Document) -> str:
    return f"""Content: {doc.page_content}
Source: {doc.metadata[FILE_NAME]}, pages: {doc.metadata[PAGE]}
"""


def create_prompt_template() -> ChatPromptTemplate:
    section = prompts_toml["retrieval_qa"]
    human_message = section["human_message"]
    system_message = section["system_message"]

    prompt_msgs = [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=system_message, input_variables=[]
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template=human_message,
                input_variables=[SUMMARIES, QUESTION, HISTORY],
            )
        ),
    ]
    return ChatPromptTemplate(messages=prompt_msgs)

class ResponseText(BaseModel):
    response: str = Field(description="The response to the user's question")
    sources: Union[str, None] = Field(
        description="The sources based on which the response was generated"
    )


def create_stuff_chain() -> LLMChain:
    class _OutputFormatter(BaseModel):
        """Output formatter. Should always be used to format your response to the user."""  # noqa: E501

        output: ResponseText  # type: ignore

    function = _OutputFormatter
    output_parser = PydanticAttrOutputFunctionsParser(
        pydantic_schema=_OutputFormatter, attr_name="output"
    )

    openai_functions = [convert_to_openai_function(f) for f in [function]]
    llm_kwargs: Dict[str, Any] = {
        "functions": openai_functions,
    }
    if len(openai_functions) == 1:
        llm_kwargs["function_call"] = {"name": openai_functions[0]["name"]}

    return LLMChain(
        llm=cfg.llm,
        prompt=create_prompt_template(),
        output_parser=output_parser,
        verbose=cfg.verbose_llm,
        llm_kwargs=llm_kwargs,
    )


qa_service_chain = create_stuff_chain()


def answer_question(question: str, history: str = "") -> ResponseText:
    if history == "":
        history = "<empty>"
    vst = init_vector_search(False)
    documents = similarity_search(vst, question)
    summaries = "\n".join([convert_document_to_text(doc) for doc in documents])
    return qa_service_chain.run(
        {SUMMARIES: summaries, QUESTION: question, HISTORY: history}
    )


if __name__ == "__main__":
    from onepoint_document_chat.log_init import logger

    question = "Which are Onepoint's projects related to the travel industry?"
    res: ResponseText = answer_question(question)
    logger.info(res)
    logger.info(type(res))
