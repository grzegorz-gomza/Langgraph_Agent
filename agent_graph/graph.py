import json
import ast
from langchain_core.runnables import RunnableLambda
from langgraph.graph import START, StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from models.openai_models import get_open_ai_json
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.agents import (
    DirectQuestionAgent,
    PlannerAgent,
    SelectorAgent,
    ReporterAgent,
    ReviewerAgent,
    RouterAgent,
    FinalReportAgent,
    EndNodeAgent,
)
from agents.agents_pdf import PDFReporterAgent

from prompts.prompts import (
    reviewer_prompt_template,
    planner_prompt_template,
    selector_prompt_template,
    reporter_prompt_template,
    router_prompt_template,
    direct_llm_prompt_template,
    final_agent_prompt_template,
    reviewer_guided_json,
    selector_guided_json,
    planner_guided_json,
    router_guided_json,
    pdf_reporter_summary_guided_json,
    direct_llm_guided_json
)
from tools.google_serper import get_google_serper
from tools.basic_scraper import scrape_website
from states.state import AgentGraphState, get_agent_graph_state, state


def create_graph(
    server=None, model=None, stop=None, model_endpoint=None, temperature=0
):
    graph = StateGraph(AgentGraphState)

    graph.add_node(
        "direct_question",
        lambda state: DirectQuestionAgent(
            state=state,
            model=model,
            server=server,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature,
        ).invoke(
            research_question=state["research_question"],
            prompt=direct_llm_prompt_template,
        ),
    )

    graph.add_node(
        "planner",
        lambda state: PlannerAgent(
            state=state,
            model=model,
            server=server,
            guided_json=planner_guided_json,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature,
        ).invoke(
            research_question=state["research_question"],
            feedback=lambda: get_agent_graph_state(
                state=state, state_key="reviewer_latest"
            ),
            prompt=planner_prompt_template,
        ),
    )

    graph.add_node(
        "selector",
        lambda state: SelectorAgent(
            state=state,
            model=model,
            server=server,
            guided_json=selector_guided_json,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature,
        ).invoke(
            research_question=state["research_question"],
            feedback=lambda: get_agent_graph_state(
                state=state, state_key="reviewer_latest"
            ),
            previous_selections=lambda: get_agent_graph_state(
                state=state, state_key="selector_all"
            ),
            serp=lambda: get_agent_graph_state(state=state, state_key="serper_latest"),
            prompt=selector_prompt_template,
        ),
    )

    graph.add_node(
        "reporter",
        lambda state: ReporterAgent(
            state=state,
            model=model,
            server=server,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature,
        ).invoke(
            research_question=state["research_question"],
            feedback=lambda: get_agent_graph_state(
                state=state, state_key="reviewer_latest"
            ),
            previous_reports=lambda: get_agent_graph_state(
                state=state, state_key="reporter_all"
            ),
            research=lambda: get_agent_graph_state(
                state=state, state_key="scraper_latest"
            ),
            prompt=reporter_prompt_template,
        ),
    )

    graph.add_node(
        "reviewer",
        lambda state: ReviewerAgent(
            state=state,
            model=model,
            server=server,
            guided_json=reviewer_guided_json,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature,
        ).invoke(
            research_question=state["research_question"],
            feedback=lambda: get_agent_graph_state(
                state=state, state_key="reviewer_all"
            ),
            reporter=lambda: get_agent_graph_state(
                state=state, state_key="reporter_latest"
            ),
            direct_question_response=state["direct_question_response"],
            pdf_report_response=lambda: get_agent_graph_state(
                state=state, state_key="pdf_report_latest"
            ),
            prompt=reviewer_prompt_template,
        ),
    )

    graph.add_node(
        "router",
        lambda state: RouterAgent(
            state=state,
            model=model,
            server=server,
            guided_json=router_guided_json,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature,
        ).invoke(
            research_question=state["research_question"],
            feedback=lambda: get_agent_graph_state(
                state=state, state_key="reviewer_all"
            ),
            prompt=router_prompt_template,
        ),
    )

    graph.add_node(
        "serper_tool",
        lambda state: get_google_serper(
            state=state,
            plan=lambda: get_agent_graph_state(state=state, state_key="planner_latest"),
        ),
    )

    graph.add_node(
        "scraper_tool",
        lambda state: scrape_website(
            state=state,
            research=lambda: get_agent_graph_state(
                state=state, state_key="selector_latest"
            ),
        ),
    )

    graph.add_node(
        "pdf_reporter",
        lambda state: PDFReporterAgent(
            state=state,
            model=model,
            server=server,
            guided_json=pdf_reporter_summary_guided_json,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature,
        ).invoke(
            research_question=state["research_question"],
            file_path=state["pdf_loaded"],
        ),
    )

    graph.add_node(
        "final_report",
        lambda state: FinalReportAgent(
                state=state,
                model=model,
                server=server,
                guided_json=None,
                stop=stop,
                model_endpoint=model_endpoint,
                temperature=temperature
            ).invoke(
            research_question=state["research_question"],
            web_report_response=lambda: get_agent_graph_state(
                state=state, state_key="reporter_latest"),
            pdf_report_response=lambda: get_agent_graph_state(
                state=state, state_key="pdf_report_latest"),
            llm_direct_response=state["direct_question_response"],
            prompt=final_agent_prompt_template
        ),
    )



    graph.add_node("end", lambda state: EndNodeAgent(state).invoke())

    # Define the edges in the agent graph
    def pass_review(state: AgentGraphState):
        review_list = state["router_response"]
        if review_list:
            review = review_list[-1]
        else:
            review = "No review"

        if review != "No review":
            if isinstance(review, HumanMessage):
                review_content = review.content
            else:
                review_content = review
            
            review_data = json.loads(review_content)
            next_agent = review_data["next_agent"]
        else:
            next_agent = "end"

        return next_agent

    def set_graph_entry_point(state: AgentGraphState, graph: StateGraph):
        if state["pdf_loaded"] == None:
            return False
        else:
            return True

    # Add edges to the graph


    # Decide if to start with llm or pdf
    graph.add_conditional_edges(
        START,
        lambda state: set_graph_entry_point(state=state, graph=graph),
        {True: "pdf_reporter", False: "direct_question"}
    )
    

    
    # llm loop
    graph.add_edge("direct_question", "reviewer")
    # pdf loop
    graph.add_edge("pdf_reporter", "reviewer")
    # web search loop
    graph.add_edge("planner", "serper_tool")
    graph.add_edge("serper_tool", "selector")
    graph.add_edge("selector", "scraper_tool")
    graph.add_edge("scraper_tool", "reporter")
    graph.add_edge("reporter", "reviewer")
    # parse review to router
    graph.add_edge("reviewer", "router")
    # decide next agent
    graph.add_conditional_edges(
        "router",
        lambda state: pass_review(state=state),
    )
    graph.add_edge("final_report", "end")

    graph.set_finish_point("end")

    return graph


def compile_workflow(graph):
    workflow = graph.compile()
    return workflow
