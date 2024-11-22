# import json
# import yaml
# import os
from termcolor import colored
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from base64 import b64decode

from models.openai_models import get_open_ai, get_open_ai_json
from models.ollama_models import OllamaModel, OllamaJSONModel
from models.vllm_models import VllmJSONModel, VllmModel
from models.groq_models import GroqModel, GroqJSONModel
from models.claude_models import ClaudModel, ClaudJSONModel
from models.gemini_models import GeminiModel, GeminiJSONModel
from prompts.prompts import (
    planner_prompt_template,
    selector_prompt_template,
    reporter_prompt_template,
    reviewer_prompt_template,
    router_prompt_template,
    pdf_reporter_prompt_template,
    direct_llm_prompt_template
)
from utils.helper_functions import get_current_utc_datetime, check_for_content, check_if_pdf_loaded
from states.state import AgentGraphState, get_agent_graph_state
from tools.pdf_extraction import pdf_extraction_tool

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

class PlannerAgent(Agent):
    def invoke(self, research_question, prompt=planner_prompt_template, feedback=None):
        feedback_value = feedback() if callable(feedback) else feedback
        feedback_value = check_for_content(feedback_value)

        planner_prompt = prompt.format(
            feedback=feedback_value,
            datetime=get_current_utc_datetime()
        )

        messages = [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": f"research question: {research_question}"}
        ]

        llm = self.get_llm()
        llm_response = llm.invoke(messages)
        llm_response_content = llm_response.content

        self.update_state("planner_response", llm_response_content)
        print(colored(f"Planner 👩🏿‍💻: {llm_response_content}", 'cyan'))
        return self.state

class SelectorAgent(Agent):
    def invoke(self, research_question, prompt=selector_prompt_template, feedback=None, previous_selections=None, serp=None):
        feedback_value = feedback() if callable(feedback) else feedback
        previous_selections_value = previous_selections() if callable(previous_selections) else previous_selections

        feedback_value = check_for_content(feedback_value)
        previous_selections_value = check_for_content(previous_selections_value)

        selector_prompt = prompt.format(
            feedback=feedback_value,
            previous_selections=previous_selections_value,
            serp=serp().content,
            datetime=get_current_utc_datetime()
        )

        messages = [
            {"role": "system", "content": selector_prompt},
            {"role": "user", "content": f"research question: {research_question}"}
        ]

        llm = self.get_llm()
        llm_response = llm.invoke(messages)
        llm_response_content = llm_response.content

        print(colored(f"selector 🧑🏼‍💻: {llm_response_content}", 'green'))
        self.update_state("selector_response", llm_response_content)
        return self.state

class ReporterAgent(Agent):
    def invoke(self, research_question, prompt=reporter_prompt_template, feedback=None, previous_reports=None, research=None):
        feedback_value = feedback() if callable(feedback) else feedback
        previous_reports_value = previous_reports() if callable(previous_reports) else previous_reports
        research_value = research() if callable(research) else research

        feedback_value = check_for_content(feedback_value)
        previous_reports_value = check_for_content(previous_reports_value)
        research_value = check_for_content(research_value)
        
        reporter_prompt = prompt.format(
            feedback=feedback_value,
            previous_reports=previous_reports_value,
            datetime=get_current_utc_datetime(),
            research=research_value
        )

        messages = [
            {"role": "system", "content": reporter_prompt},
            {"role": "user", "content": f"research question: {research_question}"}
        ]

        llm = self.get_llm(json_model=False)
        llm_response = llm.invoke(messages)
        llm_response_content = llm_response.content

        print(colored(f"Reporter 👨‍💻: {llm_response_content}", 'yellow'))
        self.update_state("reporter_response", llm_response_content)
        print(reporter_prompt)
        return self.state

class ReviewerAgent(Agent):
    def invoke(self, research_question,
                    prompt=reviewer_prompt_template,
                    reporter=None,
                    direct_question_response=None,
                    pdf_reporter_responce=None,
                    feedback=None):
        reporter_value = reporter() if callable(reporter) else reporter
        direct_question_response_value = direct_question_response() if callable(direct_question_response) else direct_question_response
        pdf_reporter_responce_value = pdf_reporter_responce() if callable(pdf_reporter_responce) else pdf_reporter_responce
        feedback_value = feedback() if callable(feedback) else feedback

        reporter_value = check_for_content(reporter_value)
        direct_question_response_value = check_for_content(direct_question_response_value)
        pdf_reporter_responce_value = check_for_content(pdf_reporter_responce_value)
        feedback_value = check_for_content(feedback_value)
        
        reviewer_prompt = prompt.format(
            reporter=reporter_value,
            direct_question_response=direct_question_response_value,
            pdf_reporter_responce=pdf_reporter_responce_value,
            state=self.state,
            feedback=feedback_value,
            datetime=get_current_utc_datetime(),
        )

        messages = [
            {"role": "system", "content": reviewer_prompt},
            {"role": "user", "content": f"research question: {research_question}"}
        ]

        llm = self.get_llm()
        llm_response = llm.invoke(messages)
        llm_response_content = llm_response.content

        print(colored(f"Reviewer 👩🏽‍⚖️: {llm_response_content}", 'magenta'))
        self.update_state("reviewer_response", llm_response_content)
        return self.state
    
class RouterAgent(Agent):
    def invoke(self, feedback=None, research_question=None, prompt=router_prompt_template):
        feedback_value = feedback() if callable(feedback) else feedback
        feedback_value = check_for_content(feedback_value)

        router_prompt = prompt.format(feedback=feedback_value)

        messages = [
            {"role": "system", "content": router_prompt},
            {"role": "user", "content": f"research question: {research_question}"}
        ]

        llm = self.get_llm()
        llm_response = llm.invoke(messages)
        llm_response_content = llm_response.content

        print(colored(f"Router 🧭: {llm_response_content}", 'blue'))
        self.update_state("router_response", llm_response_content)
        return self.state

class FinalReportAgent(Agent):
    def invoke(self, final_response=None):
        final_response_value = final_response() if callable(final_response) else final_response
        llm_response_content = final_response_value.content

        print(colored(f"Final Report 📝: {llm_response_content}", 'blue'))
        self.update_state("final_reports", llm_response_content)
        return self.state

class EndNodeAgent(Agent):
    def invoke(self):
        self.update_state("end_chain", "end_chain")
        return self.state


class PDFReporterAgent(Agent):
    def extract_pdf_elements(self, file_path: str):
        retriever = pdf_extraction_tool(file_path=file_path)
        return retriever
    
    def parse_docs(docs):
        """Split base64-encoded images and texts"""
        b64 = []
        text = []
        for doc in docs:
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        return {"images": b64, "texts": text}

    def build_prompt(kwargs):

        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.text

        # construct prompt with context (including images)
        prompt_template = pdf_reporter_prompt_template.format(
                                                            user_question=user_question,
                                                            context_text=context_text
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

        return ChatPromptTemplate.from_messages(
            [
                HumanMessage(content=prompt_content),
            ]
        )

    def invoke(self, research_question, file_path = None):
        retriever = self.extract_pdf_elements(file_path=file_path)
        llm = self.get_llm()
        
        chain = (
            {
                "context": retriever | RunnableLambda(parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(build_prompt)
            | llm
            | StrOutputParser()
        )

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

        response = chain.invoke(research_question)

        self.update_state("pdf_report_response", response)
        print(colored(f"PDFReporter Agent Response: {response}", 'green'))
        return self.state

class DirectQuestionAgent(Agent):
    def invoke(self, research_question, prompt=direct_llm_prompt_template):
        direct_llm_prompt = prompt.format(
            datetime=get_current_utc_datetime()
        )

        messages = [
            {"role": "system", "content": direct_llm_prompt},
            {"role": "user", "content": f"research question: {research_question}"}
        ]

        llm = self.get_llm(json_model=False)
        llm_response = llm.invoke(messages)
        llm_response_content = llm_response.content

        self.update_state("direct_question_response", llm_response_content)
        print(colored(f"Answer direct from LLM: {self.state['direct_question_response']}", 'light_red'))
        return self.state


# # Usage
# state = AgentGraphState()  # Assumed to be defined elsewhere
# pdf_agent = PDFReporterAgent(state=state, retriever=my_retriever, model="gpt-4o-mini", server="openai", temperature=0.5)

# response = pdf_agent.invoke(research_question="What is the impact of climate change on polar bears?", context=my_context)
