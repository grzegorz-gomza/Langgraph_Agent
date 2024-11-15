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

config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
load_config(config_path)


class PDFExtractionTool:
    def __init__(self, file_path, collection_name="multi_modal_rag"):
        self.file_path = file_path
        self.collection_name = collection_name
        self.vectorstore = Chroma(
            collection_name=self.collection_name, embedding_function=OpenAIEmbeddings()
        )
        self.store = InMemoryStore()
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore, docstore=self.store
        )

    def extract_pdf_elements(self):
        chunks = partition_pdf(
            filename=self.file_path,
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

    def separate_elements(self, chunks):
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

    def add_to_vectorstore(
        self, texts, tables, images, text_summaries, table_summaries, image_summaries
    ):
        # Add texts
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={"doc_id": doc_ids[i]})
            for i, summary in enumerate(text_summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_texts)
        self.retriever.docstore.mset(list(zip(doc_ids, texts)))

        # Add tables
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=summary, metadata={"doc_id": table_ids[i]})
            for i, summary in enumerate(table_summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_tables)
        self.retriever.docstore.mset(list(zip(table_ids, tables)))

        # Add images
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=summary, metadata={"doc_id": img_ids[i]})
            for i, summary in enumerate(image_summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_img)
        self.retriever.docstore.mset(list(zip(img_ids, images)))

    def retrieve_response(self, query):
        # Retrieve relevant information
        docs = self.retriever.invoke(query)
        return docs


# This function can be put into a utility file if needed
def display_base64_image(base64_code):
    import base64
    from IPython.display import Image, display

    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))

