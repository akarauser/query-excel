import pandas as pd
import streamlit as st
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_ollama import ChatOllama

from .utils._logger import logger
from .utils._validation import config_args

st.set_page_config(page_title="Query Excel")

# Session States
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "context" not in st.session_state:
    st.session_state["context"] = []

# System Prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are a helpful AI assistant who answers user questions based on the provided context, without generating any HTML code."""
)

# Prompt
HUMAN_PROMPT = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don"t know".
            ### Context:
            {context}

            ### Question:
            {question}

            ### Answer:"""

chat_template = ChatPromptTemplate(
    messages=[system_prompt, HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)]
)

# LLM
try:
    LLM = ChatOllama(model=config_args.base_model, base_url=config_args.local_url)
except Exception as e:
    logger.error(f"Error initializing ChatOllama: {e}")
    LLM = None

# Output Parser
output_parser = StrOutputParser()

# Chain
qna_chain = chat_template | LLM | output_parser


# Helper Function
def llm_stream(context, question):
    """
    Initiliaze the LLM and stream the results.
    """
    try:
        if LLM is None:
            yield "LLM initialization failed.  Cannot answer questions."
            return

        for event in qna_chain.stream({"context": context, "question": question}):
            yield event
    except Exception as e:
        logger.error(f"Error streaming from LLM: {e}")
        yield "An error occurred while generating the answer."


# File Uploader
with st.sidebar:
    excel_doc = st.file_uploader(
        "Upload your Excel file (xlsx)", type=["xlsx"], key="excel_doc"
    )
    if excel_doc:
        try:
            pd_doc = pd.read_excel(excel_doc)
            pd_doc.to_excel(config_args.temporary_file_path)
            loader = UnstructuredExcelLoader(
                config_args.temporary_file_path, mode="elements"
            )
            excel = loader.load()
            context = excel[0].metadata["text_as_html"]
            st.session_state.context = context
            st.success("Done!  Ready to ask questions.")
            logger.info("Document uploaded succesfuly.")

        except Exception as e:
            st.error(f"Error processing Excel file: {e}")
            logger.error(f"Error processing Excel file: {e}")
            st.warning("Please ensure your excel file is a valid .xlsx file.")


# Conversation Logic
def conversation():
    """
    Inference for chat.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Type here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            message = st.write_stream(llm_stream(st.session_state.context, prompt))
            st.session_state.messages.append({"role": "assistant", "content": message})


conversation()
