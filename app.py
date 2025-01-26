import operator
from typing import *
import requests
import tempfile
import os
import streamlit as st
import asyncio

# LangChain
from langchain.tools import BaseTool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

# LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool

# Plotly
import plotly.graph_objects as go
from plotly.io import from_json

# Show title and description.
st.title("🤖 ChatWithYourData")
st.write("""
🧠 This data assistant lets you interact with your database using natural language.
🚀 Powered by Streamlit and Langgraph, it enables easy querying of a database through a conversational interface — all without needing to write SQL. 
🔎 Perfect for exploring and interacting with your data more intuitively.
         
---
""")

class HumanInputStreamlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name: str = "human"
    description: str = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )

    def _run(self, query: str, run_manager=None) -> str:
        """Use the Human input tool."""
        return st.text_input("Agent question:", query)

    async def _arun(self, query: str, run_manager=None) -> str:
        """Use the Human input tool."""
        return self._run(query)

class ChatState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], operator.add]

def format_messages_for_bedrock(messages):
    formatted_messages = []
    for i, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            formatted_messages.append({"role": "system", "content": message.content})
        elif isinstance(message, HumanMessage):
            formatted_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            formatted_messages.append({"role": "assistant", "content": message.content})
        
        # Ensure alternating user/assistant messages
        if i > 0 and formatted_messages[-1]["role"] == formatted_messages[-2]["role"] == "assistant":
            formatted_messages.insert(-1, {"role": "user", "content": "Please continue."})
    
    return formatted_messages

def initialize_session():
    if 'runnable' not in st.session_state:

        llm = AzureChatOpenAI(base_url=st.secrets["LLM_BASE_URL"],
            openai_api_version=st.secrets["AI_VERSION"],
            api_key = st.secrets["API_SERVICE_KEY"],
            temperature = 0.0,
            #model_kwargs={"stop": ["\nObservation", "Observation"]} # this should further prevent halucinations
        )        
        tools = [
            HumanInputStreamlit(),
            plotly_tool,
        ]
        memory = MemorySaver()
        agent = create_react_agent(llm, tools, checkpointer=memory)

        st.session_state.runnable = agent
        st.session_state.state = ChatState(messages=[])

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def main():
    initialize_session()

    # Display chat history
    for message in st.session_state.state['messages']:
        if isinstance(message, HumanMessage):
            st.chat_message("human").write(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").write(message.content)
        elif isinstance(message, go.Figure):
            st.plotly_chart(message, use_container_width=True)

    # Get user input
    user_input = st.chat_input("Please type your message here")

    if user_input:
        # Display the new user message immediately
        st.chat_message("human").write(user_input)

        # Append the new message to the state
        st.session_state.state['messages'] += [HumanMessage(content=user_input)]


        state_for_llm = st.session_state.state

        # Create placeholders for the assistant's response and agent questions
        assistant_placeholder = st.empty()
        agent_question_placeholder = st.empty()

        full_response = ""
        agent_question = ""
        figure = None

        async def process_stream():
            nonlocal full_response, agent_question, figure
            async for event in st.session_state.runnable.astream_events(
                state_for_llm,
                version="v1",
                config={'configurable': {'thread_id': 'thread-1'}}
            ):
                if event["event"] == "on_tool_end":
                    if event["data"].get("output") and event["data"].get("output").artifact:
                        figure = from_json(event["data"].get("output").artifact)
                if event["event"] == "on_chat_model_stream":
                    c = event["data"]["chunk"].content
                    if c and len(c) > 0 and isinstance(c[0], dict) and c[0]["type"] == "text":
                        content = c[0]["text"]
                    elif isinstance(c, str):
                        content = c
                    else:
                        content = ""
                    full_response += content
                    with assistant_placeholder.container():
                        st.markdown(full_response + "▌")
                elif event["event"] == "on_tool_start":
                    tool_name = event["name"]
                    tool_input = event["data"]["input"]
                    if tool_name == "human":
                        agent_question = tool_input
                        with agent_question_placeholder.container():
                            st.info(f"Agent question: {agent_question}")
                            user_answer = st.text_input("Your answer:")
                            if user_answer:
                                return user_answer

        while True:
            user_answer = run_async(process_stream())
            if user_answer:
                st.session_state.state['messages'] += [HumanMessage(content=user_answer)]
            else:
                break

        # Update the assistant's response
        with assistant_placeholder.container():
            st.markdown(full_response)
            if figure:
                st.plotly_chart(figure, use_container_width=True)

        # Append the assistant's response to the state
        st.session_state.state['messages'] += [AIMessage(content=full_response)]
        #if figure:
        #    st.session_state.state["messages"] += [figure]

def render_plotly_graph(figureCode: str):
    local_vars = {}
    exec(figureCode, {"go": go}, local_vars)
    fig = local_vars.get("fig")
    return "generated a chart from the provided figure", fig.to_json()

plotly_tool = Tool(
    name = "plotly_tool",
    func=render_plotly_graph,
    description="useful for when you need to render a graph using plotly; the figureCode must only import plotly.graph_objects as go and must provide a local variable named fig as a result",
    response_format='content_and_artifact'
)

if __name__ == "__main__":
    main()