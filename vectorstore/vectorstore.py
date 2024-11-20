import os
import uuid
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

from utils.helper_functions import load_config

config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
load_config(config_path)

def create_vectorstore(id_key="doc_id"):
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = id_key

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    return retriever

def add_to_vectorstore(texts, tables, images, text_summaries, table_summaries, image_summaries, retriever, id_key="doc_id"):
    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images)))

    return retriever

def retrieve_response(query, retriever):
    # Retrieve relevant information
    docs = retriever.invoke(query)
    return docs 