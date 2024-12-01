import streamlit as st
import os
import json
import yaml
import tempfile

import sys
import os

# Adding the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from agent_graph.graph import create_graph, compile_workflow

def update_config(
    serper_api_key,
    openai_llm_api_key,
    groq_llm_api_key,
    claud_llm_api_key,
    gemini_llm_api_key,
):
    config_path = os.path.join(os.getcwd(), "config", "config.yaml")

    # reasure that the 'config' directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Read existing config or create a new one
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file) or {}
    else:
        config = {}

    config["SERPER_API_KEY"] = serper_api_key
    config["OPENAI_API_KEY"] = openai_llm_api_key
    config["GROQ_API_KEY"] = groq_llm_api_key
    config["CLAUDE_API_KEY"] = claud_llm_api_key
    config["ANTHROPIC_API_KEY"] = claud_llm_api_key
    config["GEMINI_API_KEY"] = gemini_llm_api_key

    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key
    if openai_llm_api_key:
        os.environ["OPENAI_API_KEY"] = openai_llm_api_key
    if groq_llm_api_key:
        os.environ["GROQ_API_KEY"] = groq_llm_api_key
    if claud_llm_api_key:
        os.environ["CLAUDE_API_KEY"] = claud_llm_api_key
        os.environ["ANTHROPIC_API_KEY"] = claud_llm_api_key
    if gemini_llm_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_llm_api_key

    with open(config_path, "w") as file:
        yaml.safe_dump(config, file)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("Configuration updated successfully.")


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

        for event in self.workflow.stream(dict_inputs, limit):

            if "final_report" in event.keys():
                final_answer = event["final_report"]["final_reports"]
                return final_answer


        return "Workflow did not reach final report"


def main():
    st.set_page_config(page_title="Langgraph AI Agent", page_icon=":speech_balloon:")

    # Initialize Workflow of the app
    if "chat_workflow" not in st.session_state:
        st.session_state.chat_workflow = ChatWorkflow()
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Initialize PDF file state
    if "pdf_loaded" not in st.session_state:
        st.session_state.pdf_loaded = ""

    # Left sidebar with file upload and settings
    with st.sidebar:
        st.header("Application settings")
        ### PDF ###
        with st.expander("Upload your PDF file"):
            # Define header
            st.subheader("Your documents")
            # create a temporary file
            tmp_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
            # File Uploader on the sidebar
            pdf_doc = st.file_uploader(
                "Choose a PDF file", type="pdf", accept_multiple_files=False,
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
                    except Exception as e:
                        st.error("PDF upload failed", icon="❌")

        ### CHAT SETTINGS ###
        with st.expander("Chat settings"):
            # st.subheader("Chat settings")

            server = st.selectbox(
                "Choose the server: ",
                ["openai"], # "ollama", "vllm", "groq", "claude", "gemini"],
                key="server",
            )
            st.caption("Currently only OpenAI is supported.",)
            st.html("<br>")

            model = st.selectbox(
                "Choose the model: ",
                ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4", "gpt-4o-mini", "gpt-4o"],
                key="llm_model",
            )
            st.html("<br>")

            api_key = st.text_input("Give your API Key:", key="api_key")
            st.caption("get your API key from https://platform.openai.com/account/api-keys")
            st.html("<br>")

            recursion_limit = st.number_input(
                "Recursion limit:", min_value=1, value=20, max_value=50, key="recursion_limit"
            )
            st.caption("Recursion limit is the maximum number of agent actions the workflow will take before stopping.")
            st.html("<br>")

            serper_api_key = st.text_input(
                "Give your Serper API Key:", key="serper_api_key"
            )
            st.caption("Serper is a tool for web search. Get your API key from https://serper.dev/")
            st.html("<br>")

            model_endpoint = None
            # st.text_input(
            #     "Model endpoint vLLM (optional):", key="server_endpoint"
            # )

            stop = "<|end_of_text|>"

            temperature = st.slider("Temperature:", 0.0, 1.0, 0.0, 0.05, key="temperature")
            st.caption("Temperature is a parameter that controls the randomness of the output. A higher temperature will make the output more random, while a lower temperature will make the output more deterministic.")

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

        with st.expander("Instructions to use the Aplication"):
            st.markdown(
                """
                ## Instructions to use the Aplication
                1. Choose your server and LLM
                2. Give the API key for your server and serpenter
                3. Build the workflow by submitting the settings
                4. Optional: upload a PDF file
                5. Optional: set the recursion limit
                6. Ask a question in the text bar
                7. The application will respond to your question
                """
            )

        with st.expander("Disclaimer about the Aplication"):
            st.markdown(
                """
                ## Disclaimer

                **This application was created for learning purposes and is not intended for real-world use.**

                This means that the application is not bug-free and may crash.
                The agent may feel a bit slow because it uses multiple AI agents to determine the correct answer.
                Large Language Models (LLMs) are trained on specific datasets available at the time of their training.
                They are not aware of the current state of the world. To address this limitation, this agent uses a workflow that either relies on the provided context from a given PDF file or attempts to find the necessary information on the web.
                The AI agents "communicate" with one another, passing along information from previous agents. Over time, the information passed to the next agent becomes extensive, which increases the processing time for an LLM.
                Nevertheless, the application can provide answers based on the information contained in the PDF file or, when necessary, search the web for answers.

                **This application is a proof of concept and is not intended for production use.**
                It is provided "as is" without any warranties or guarantees of any kind.
                Use at your own risk.

                **About the project and developer:**

                GitHub: https://github.com/grzegorz-gomza/Langgraph_Agent

                LinkedIn: https://www.linkedin.com/in/gregory-gomza/
                """
            )
    # Right sidebar with instructions
    st.title("AI Chatbot")
    st.subheader("Builded with Langgraph by Gregor Gomza")
    # Main content
    # col1, col2 = st.columns([1, 2])

    # with col2:

    # Display chat messages from history on app rerun
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.markdown(message[-1])
        else:
            with st.chat_message("assistant"):
                st.markdown(message[-1])

    user_message = st.chat_input("Your Query:", key="user_message")

    if user_message:
        if not st.session_state.chat_workflow.workflow:
            st.error("Workflow was not built yet. Please set up the environment.")
        else:
            # Get user input *before* updating session state
            user_input = user_message
            st.session_state.chat_history.append(("You", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.spinner("Waiting for response..."):
                with st.chat_message("assistant"):
                    response = st.session_state.chat_workflow.invoke_workflow(user_input)
                    st.session_state.chat_history.append(("AI", response))
                    st.markdown(response)

    # # Display chat history
    # if st.session_state.chat_history:
    #     for human, ai in st.session_state.chat_history:
    #         with st.chat_message("user"):
    #             st.markdown(prompt)
    #         st.markdown(f"**{speaker}:** {message}")


if __name__ == "__main__":
    main()
