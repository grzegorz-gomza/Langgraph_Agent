from termcolor import colored
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prompts.prompts import (
    pdf_text_summary_prompt_template,
    pdf_table_summary_prompt_template,
    pdf_image_summary_prompt_template,
)
from utils.helper_functions import get_current_utc_datetime
from states.state import AgentGraphState

class Agent:
    def __init__(self, state: AgentGraphState, model=None, server=None, temperature=0, model_endpoint=None, stop=None, guided_json=None):
        self.state = state
        self.model = model
        self.server = server
        self.temperature = temperature
        self.model_endpoint = model_endpoint
        self.stop = stop
        self.guided_json = guided_json

    def get_llm(self, json_model=True):
        if self.server == 'openai':
            return get_open_ai_json(model=self.model, temperature=self.temperature) if json_model else get_open_ai(model=self.model, temperature=self.temperature)
        if self.server == 'ollama':
            return OllamaJSONModel(model=self.model, temperature=self.temperature) if json_model else OllamaModel(model=self.model, temperature=self.temperature)
        if self.server == 'vllm':
            return VllmJSONModel(
                model=self.model, 
                guided_json=self.guided_json,
                stop=self.stop,
                model_endpoint=self.model_endpoint,
                temperature=self.temperature
            ) if json_model else VllmModel(
                model=self.model,
                model_endpoint=self.model_endpoint,
                stop=self.stop,
                temperature=self.temperature
            )
        if self.server == 'groq':
            return GroqJSONModel(
                model=self.model,
                temperature=self.temperature
            ) if json_model else GroqModel(
                model=self.model,
                temperature=self.temperature
            )
        if self.server == 'claude':
            return ClaudJSONModel(
                model=self.model,
                temperature=self.temperature
            ) if json_model else ClaudModel(
                model=self.model,
                temperature=self.temperature
            )
        if self.server == 'gemini':
            return GeminiJSONModel(
                model=self.model,
                temperature=self.temperature
            ) if json_model else GeminiModel(
                model=self.model,
                temperature=self.temperature
            )      

    def update_state(self, key, value):
        self.state = {**self.state, key: value}

class TextSummaryAgent(Agent):
    def invoke(self, extracted_text, prompt=pdf_text_summary_prompt_template):
        text_summary_prompt = prompt.format(
            datetime=get_current_utc_datetime()
        )

        messages = ChatPromptTemplate.from_template(text_summary_prompt)

        llm = self.get_llm()
        summarize_chain = {"extracted_text": lambda x: x} | messages | llm | StrOutputParser()
        text_summaries = summarize_chain.batch(extracted_text, {"max_concurrency": 3})

        print(colored(f"Text Summary : {text_summaries}", 'red'))
        return text_summaries


class TableSummaryAgent(Agent):
    def invoke(self, extracted_table, prompt=pdf_table_summary_prompt_template):
        table_summary_prompt = prompt.format(
            datetime=get_current_utc_datetime()
        )

        messages = ChatPromptTemplate.from_template(table_summary_prompt)

        llm = self.get_llm()
        summarize_chain = {"extracted_table": lambda x: x} | messages | llm | StrOutputParser()
        tables_summaries = summarize_chain.batch(extracted_table, {"max_concurrency": 3})

        print(colored(f"Table Summary : {tables_summaries}", 'Yellow'))
        return tables_summaries

class ImageSummaryAgent(Agent):
    def invoke(self, extracted_images, prompt=pdf_image_summary_prompt_template):
        image_summary_prompt = prompt.format(
            datetime=get_current_utc_datetime()
        )
        
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

        llm = self.get_llm()
        summarize_chain = prompt | llm | StrOutputParser()
        images_summaries = summarize_chain.batch(extracted_images)

        print(colored(f"Table Summary : {images_summaries}", 'Pink'))
        return images_summaries
