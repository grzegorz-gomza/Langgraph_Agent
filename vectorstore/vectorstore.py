import os
import uuid
import faiss
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

from utils.helper_functions import load_config

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
load_config(config_path)

class VectorStoreManager:
    def __init__(self, id_key="doc_id"):
        """
        Initialize the VectorStoreManager instance.
        """
        self.id_key = id_key
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self):
        """
        Creates the retriever with an empty FAISS vectorstore and an in-memory store.
        """
        # Create an empty vectorstore
        embeddings = OpenAIEmbeddings()
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

        self.vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        # Create an in-memory store for documents
        store = InMemoryStore()
        # Create the retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=store,
            id_key=self.id_key
        )
        print("Vectorstore created successfully.")

    def add_to_vectorstore(self, texts, tables, images, text_summaries, table_summaries, image_summaries):
        """
        Adds documents (texts, tables, images) along with their summaries to the vectorstore.
        """
        try:
            if not texts and not tables and not images:
                raise ValueError("Nothing to store in Vectorstore")

            print("Adding documents to vectorstore...")
            # Add text data
            if all((len(texts) > 0, len(text_summaries) > 0)):
                doc_ids = [str(uuid.uuid4()) for _ in texts]
                summary_texts = [
                    Document(page_content=summary, metadata={self.id_key: doc_ids[i]}) 
                    for i, summary in enumerate(text_summaries)
                ]
                self.vectorstore.add_documents(summary_texts)
                self.retriever.docstore.mset(list(zip(doc_ids, texts)))
                print("Texts were successfully added to vectorstore.")
            else:
                print("No texts to add to vectorstore.")

            # Add table data
            if all((len(tables) > 0, len(table_summaries) > 0)):
                table_ids = [str(uuid.uuid4()) for _ in tables]
                summary_tables = [
                    Document(page_content=summary, metadata={self.id_key: table_ids[i]}) 
                    for i, summary in enumerate(table_summaries)
                ]
                self.vectorstore.add_documents(summary_tables)
                self.retriever.docstore.mset(list(zip(table_ids, tables)))
                print("Tables were successfully added to vectorstore.")
            else:
                print("No tables to add to vectorstore.")
                
            # Add image data
            if all((len(images) > 0, len(image_summaries) > 0)):
                img_ids = [str(uuid.uuid4()) for _ in images]
                summary_images = [
                    Document(page_content=summary, metadata={self.id_key: img_ids[i]}) 
                    for i, summary in enumerate(image_summaries)
                ]
                self.vectorstore.add_documents(summary_images)
                self.retriever.docstore.mset(list(zip(img_ids, images)))
                print("Images were successfully added to vectorstore.")
            else:
                print("No images to add to vectorstore.")
            print("All documents were added to vectorstore successfully.")
        
        except Exception as e:
            print(f"Error occurred while adding documents to vectorstore: {e}")

    def get_runnable_retriever(self):
        if self.retriever is None:
            raise ValueError("Retriever has not been created. Please run create_vectorstore() first.")  # Throw error if retriever is missing
        print("Retriever type:", type(self.retriever))
        return self.retriever #.as_retriever()  # Convert retriever to a Runnable-compatible object
