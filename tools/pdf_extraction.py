import os
import uuid
from unstructured.partition.pdf import partition_pdf
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.output_parsers import StrOutputParser

from utils.helper_functions import load_config
from vectorstore.vectorstore import create_vectorstore, add_to_vectorstore, retrieve_response
from agents.agents import (
                            AgentGraphState,
                            PDFReporterAgent,
                            TextSummaryAgent,
                            PDFTableSummaryAgent,
                            PDFImageSummaryAgent
                            )

config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
load_config(config_path)


        

def extract_pdf_elements(file_path):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    return chunks

def separate_elements(chunks):
    tables = []
    texts = []
    images = []

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
            elements = chunk.metadata.orig_elements
            for el in elements:
                if "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)
    return texts, tables, images


# This function can be put into a utility file if needed
def display_base64_image(base64_code):
    import base64
    from IPython.display import Image, display

    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))


def pdf_extraction_tool(file_path: str):
    chunks = extract_pdf_elements(file_path) # Extract elements from PDF
    texts, tables, images = separate_elements(chunks) # Separate elements into text, tables and images

    # Summarize data
    text_summaries = TextSummaryAgent.invoke(texts)
    table_summaries = PDFTableSummaryAgent.invoke(tables)
    image_summaries = PDFImageSummaryAgent.invoke(images)

    # Create vectorstore
    retriever = create_vectorstore()
    
    # Add data to vectorstore
    retriever = add_to_vectorstore(texts, tables, images, text_summaries, table_summaries, image_summaries, retriever)

    return retriever

