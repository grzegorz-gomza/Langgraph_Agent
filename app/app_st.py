import streamlit as st
import os
import json
import yaml
import tempfile

from agent_graph.graph import create_graph, compile_workflow

def update_config(
    serper_api_key,
    openai_llm_api_key,
    groq_llm_api_key,
    claud_llm_api_key,
    gemini_llm_api_key,
):
    config_path = os.path.join(os.getcwd(), "config", "config.yaml")

    # Upewnij się, że katalog 'config' istnieje
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Wczytaj istniejącą konfigurację lub utwórz nową
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file) or {}
    else:
        config = {}

    config["SERPER_API_KEY"] = serper_api_key
    config["OPENAI_API_KEY"] = openai_llm_api_key
    config["GROQ_API_KEY"] = groq_llm_api_key
    config["CLAUD_API_KEY"] = claud_llm_api_key
    config["GEMINI_API_KEY"] = gemini_llm_api_key

    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key
    if openai_llm_api_key:
        os.environ["OPENAI_API_KEY"] = openai_llm_api_key
    if groq_llm_api_key:
        os.environ["GROQ_API_KEY"] = groq_llm_api_key
    if claud_llm_api_key:
        os.environ["CLAUD_API_KEY"] = claud_llm_api_key
    if gemini_llm_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_llm_api_key

    # with open(config_path, "w") as file:
    #     yaml.safe_dump(config, file)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("Konfiguracja zaktualizowana pomyślnie.")


class ChatWorkflow:
    def __init__(self):
        self.workflow = None
        self.recursion_limit = 40

    def build_workflow(
        self, server, model, model_endpoint, temperature, recursion_limit=40, stop=None
    ):
        graph = create_graph(
            server=server,
            model=model,
            model_endpoint=model_endpoint,
            temperature=temperature,
            stop=stop,
        )
        self.workflow = compile_workflow(graph)
        self.recursion_limit = recursion_limit

    def invoke_workflow(self, message_content):
        if not self.workflow:
            return "Workflow has not been built yet. Please update settings first."

        dict_inputs = {"research_question": message_content,
                        "pdf_loaded": st.session_state.pdf_loaded if st.session_state.pdf_loaded else None}
        limit = {"recursion_limit": self.recursion_limit}
        reporter_state = None
        for event in self.workflow.stream(dict_inputs, limit):
            next_agent = ""
            if "router" in event.keys():
                state = event["router"]
                reviewer_state = state["router_response"]
                reviewer_state_dict = json.loads(reviewer_state)
                next_agent_value = reviewer_state_dict["next_agent"]
                if isinstance(next_agent_value, list):
                    next_agent = next_agent_value[-1]
                else:
                    next_agent = next_agent_value

            if next_agent == "final_report":
                router_responce = event["router"]["router_response"]
                llm_responce = state["direct_question_response"]
                if isinstance(router_responce, list) and router_responce != []:
                    output = router_responce[-1]
                    return output.content if output else "No report available"
                else:
                    return llm_responce[-1].content if llm_responce else "No answer available"
                # state = event["router"]
                # reporter_state = state["reporter_response"]
                # if isinstance(reporter_state, list) and reporter_state != []:
                #     reporter_state = reporter_state[-1]
                #     return (
                #         reporter_state.content if reporter_state else "No report available"
                #     )
                # else:
                #     return state["direct_question_response"].content

        return "Workflow did not reach final report"


def main():
    st.set_page_config(page_title="Langgraph AI Agent", page_icon=":speech_balloon:")

    # Inicjalizacja stanu aplikacji
    if "chat_workflow" not in st.session_state:
        st.session_state.chat_workflow = ChatWorkflow()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "pdf_loaded" not in st.session_state:
        st.session_state.pdf_loaded = ""

    # Pasek boczny z ustawieniami
    with st.sidebar:
        # Define header
        st.subheader("Your documents")
        # create a temporary file
        tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        # File Uploader on the sidebar
        pdf_doc = st.file_uploader(
            "Choose a PDF file", type="pdf", accept_multiple_files=False
        )

        # Create the button to upload the file
        if st.button("Upload PDF"):
            with st.spinner("Uploading PDF..."):
                try:
                    if pdf_doc is not None:
                        # Save the PDF file temporarily
                        tmp_file.write(pdf_doc.getvalue())
                        tmp_file.flush()
                        # Pass the path of the temporary file to session state
                        st.session_state.pdf_loaded = tmp_file.name
                        tmp_file.close()
                    st.success("PDF uploaded successfully", icon="✅")
                    print("path:", st.session_state.pdf_loaded)
                except Exception as e:
                    st.error("PDF upload failed", icon="❌")
                
                st.subheader("Chat settings")

        server = st.selectbox(
            "Choose the server:",
            ["openai", "ollama", "vllm", "groq", "claude", "gemini"],
            key="server",
        )

        recursion_limit = st.number_input(
            "Recursion limit:", min_value=1, value=5, key="recursion_limit"
        )

        api_key = st.text_input("Give your API Key:", key="api_key")
        model = st.text_input("LLM name:", key="llm_model", value="gpt-4o-mini") # gpt-3.5-turbo
        serper_api_key = st.text_input(
            "Give your Serper API Key:", key="serper_api_key"
        )
        model_endpoint = st.text_input(
            "Model endpoint vLLM (optional):", key="server_endpoint"
        )
        stop = st.text_input("Stop token:", value="<|end_of_text|>", key="stop_token")
        temperature = st.slider("Temperature:", 0.0, 1.0, 0.0, 0.05, key="temperature")

        if st.button("Update settings"):
            update_config(
                serper_api_key=serper_api_key,
                openai_llm_api_key=api_key if server == "openai" else None,
                groq_llm_api_key=api_key if server == "groq" else None,
                claud_llm_api_key=api_key if server == "claude" else None,
                gemini_llm_api_key=api_key if server == "gemini" else None,
            )

            st.session_state.chat_workflow.build_workflow(
                server, model, model_endpoint, temperature, recursion_limit, stop
            )
            st.success(f"Workflow was created successfully for server '{server}'.")

    # Główna część aplikacji - Chat
    st.title("Langgraph AI Agent")

    # Układ kolumn - lewa kolumna pusta, prawa z czatem
    col1, col2 = st.columns([1, 2])

    with col2:
        if "user_message" not in st.session_state:
            st.session_state.user_message = ""

        user_message = st.text_input("Your Query:", key="user_message")

        if st.button("Send"):
            if not st.session_state.chat_workflow.workflow:
                st.error("Workflow was not built yet. Please set up the environment.")
            else:
                # Get user input *before* updating session state
                user_input = user_message
                st.session_state.chat_history.append(("You", user_input))
                response = st.session_state.chat_workflow.invoke_workflow(user_input)
                st.session_state.chat_history.append(("AI", response))

        # Wyświetl historię chatu
        if st.session_state.chat_history:
            for speaker, message in st.session_state.chat_history:
                st.markdown(f"**{speaker}:** {message}")


if __name__ == "__main__":
    main()
