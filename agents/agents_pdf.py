from termcolor import colored
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

import os
import uuid
import json
from base64 import b64decode
from unstructured.partition.pdf import partition_pdf

from prompts.prompts import (
    pdf_text_summary_prompt_template,
    pdf_table_summary_prompt_template,
    pdf_image_summary_prompt_template,
    pdf_reporter_prompt_template
)
from utils.helper_functions import get_current_utc_datetime, load_config
from states.state import AgentGraphState
from agents.agents import Agent

from vectorstore.vectorstore import VectorStoreManager

config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
load_config(config_path)

class PDFReporterAgent(Agent):
    def __init__(self,retriever=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = retriever

    def extract_pdf_elements(self, file_path):
        chunks = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=5000,
            combine_text_under_n_chars=1000,
            new_after_n_chars=3000,
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

    def display_base64_image(self, base64_code):
        import base64
        from IPython.display import Image, display

        image_data = base64.b64decode(base64_code)
        display(Image(data=image_data))

    def summarize_text(self, llm, extracted_text, prompt=pdf_text_summary_prompt_template):
        if not extracted_text:  # Validate and handle empty input
            return []

        if isinstance(extracted_text, str):  # Ensure list format for batch processing
            extracted_text = [extracted_text]

        current_datetime = get_current_utc_datetime()
        text_summary_prompt = prompt + f"\n\nCurrent date and time: {current_datetime}"


        messages = ChatPromptTemplate.from_template(text_summary_prompt)
        summarize_chain = {"extracted_text": lambda x: x} | messages | llm | StrOutputParser()
        text_summaries = summarize_chain.batch(extracted_text, {"max_concurrency": 3})
        return text_summaries

    def summarize_table(self, llm, extracted_table, prompt=pdf_table_summary_prompt_template):
        if not extracted_table:
            return []
        current_datetime = get_current_utc_datetime()
        table_summary_prompt = prompt + f"\n\nCurrent date and time: {current_datetime}"

        messages = ChatPromptTemplate.from_template(table_summary_prompt)

        summarize_chain = {"extracted_table": lambda x: x} | messages | llm | StrOutputParser()
        tables_summaries = summarize_chain.batch(extracted_table, {"max_concurrency": 3})

        print(colored(f"Table Summary : {tables_summaries}", 'Yellow'))
        return tables_summaries

    def summarize_image(self, llm, extracted_images, prompt=pdf_image_summary_prompt_template):
        if not extracted_images:
            return []
        
        current_datetime = get_current_utc_datetime()
        image_summary_prompt = prompt + f"\n\nCurrent date and time: {current_datetime}"

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": image_summary_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]

        prompt = ChatPromptTemplate.from_messages(messages)

        summarize_chain = prompt | llm | StrOutputParser()
        images_summaries = summarize_chain.batch(extracted_images)

        print(colored(f"Image Summary : {images_summaries}", 'Pink'))
        return images_summaries

    def pdf_extraction_tool(self, file_path: str):
        llm = self.get_llm()
        chunks = self.extract_pdf_elements(file_path)  # Extract elements from PDF
        texts, tables, images = self.separate_elements(chunks)  # Separate elements into text, tables, and images
        
        # Summarize data
        text_summaries = self.summarize_text(llm, texts)
        table_summaries = self.summarize_table(llm, tables)
        image_summaries = self.summarize_image(llm, images)
        
        # Use the VectorStoreManager
        vectorstore_manager = VectorStoreManager()
        vectorstore_manager.create_vectorstore()  # Create the vectorstore
        vectorstore_manager.add_to_vectorstore(
            texts, tables, images, 
            text_summaries, table_summaries, image_summaries
        )  # Add data to the vectorstore
        
        print(colored(f"Retriever created", 'green'))
        return vectorstore_manager

    def create_retriever(self, file_path=None):
        if self.retriever is None and file_path is not None:
            vectorstore_manager = self.pdf_extraction_tool(file_path=file_path)  # Build vector store
            self.retriever = vectorstore_manager.get_runnable_retriever()
            print("PDFReporter Agent: Retriever created")
            return self.retriever
        elif self.retriever is not None and file_path is not None:
            return self.retriever
        else:
            print("Retriever has not been created. Please run create_vectorstore() first.")
        
    def parse_docs(self, docs):
        """Split base64-encoded images and texts"""
        b64 = []
        text = []
        for doc in docs:
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception as e:
                text.append(doc)
        return {"images": b64, "texts": text}

    def build_prompt(self, kwargs):

        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.text

        # construct prompt with context (including images)
        prompt_template = pdf_reporter_prompt_template.format(
                                                            question=user_question,
                                                            context_text=context_text,
                                                            datetime=get_current_utc_datetime()
                                                            )

        prompt_content = [{"type": "text", "text": prompt_template}]

        if len(docs_by_type["images"]) > 0:
            for image in docs_by_type["images"]:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )

        prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessage(content=prompt_content),
            ]
        )
        return prompt

    def debug_input(self, input_dict):
    # Logujemy wartości context oraz question
        print("\nDEBUG: Przekazane do modelu:")
        print("Context:", input_dict.get("context"))
        print("Question:", input_dict.get("question"))
        return input_dict  # Ważne, aby zwrócić dane w niezmienionej postaci

    def invoke(self, research_question, file_path = None):
        retriever = self.create_retriever(file_path)
        if retriever:
            llm = self.get_llm()
            print("PDFReporter Agent: LLM created")
        # Chain setup with context, processing, and final response retrieval
            chain = (
                {
                    "context": retriever | RunnableLambda(self.parse_docs),  # Make retriever compatible using .as_retriever()
                    "question": RunnablePassthrough(),
                }
                # | RunnableLambda(self.debug_input)  # Debuging
                | RunnableLambda(self.build_prompt)  # Format the prompt
                | llm  # Pass the prompt to the language model
                | StrOutputParser()  # Parse the model output
            )
            print("PDFReporter Agent: Chain created")

            # chain_with_sources = {
            #     "context": retriever | RunnableLambda(parse_docs),
            #     "question": RunnablePassthrough(),
            # } | RunnablePassthrough().assign(
            #     response=(
            #         RunnableLambda(build_prompt)
            #         | llm
            #         | StrOutputParser()
            #     )
            # )
            
        response = chain.invoke(research_question)  # Execute the chain with the research question as input
        if isinstance(response, HumanMessage):
            responce_content = json.loads(json.dumps(response.content, ensure_ascii=False, indent=4))
        else:
            responce_content = json.loads(json.dumps(response, ensure_ascii=False, indent=4))
        self.update_state("pdf_report_response", responce_content)  # Update the internal state
        print(colored(f"PDFReporter Agent Response: {responce_content} \nType: {type(responce_content)}", 'green'))  # Output the response
        return self.state
