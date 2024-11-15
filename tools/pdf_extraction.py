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


def pdf_extraction_tool(state: AgentGraphState, file_path: str, user_query: str):
    chunks = extract_pdf_elements(file_path)
    texts, tables, images = separate_elements(chunks)
# zrob tego toola jako jeden pelen workflow od przyjecia pliku pdf, po stworzenie vectorstore zapisanie wszystkiego 
# do vectorstore az po wyplucie summary co do zadanego pytania. Zrob logike, ktora nie wymaga podwojnego
# ladowania pliku pdf.
    state = {**state, "pdf_extracted_tex": results}
        return state
