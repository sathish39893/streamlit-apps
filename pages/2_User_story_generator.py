
from __future__ import annotations
import streamlit as st
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser


# This chatbot generates a user story in gherkin format.
# - Input the application name and user role to startwith.
# - Add scenarios to generate user stories

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="langchain_search_api_key_openai", type="password"
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    model_name = st.selectbox("Select a model",
        ("gpt-4o","gpt-3.5-turbo"),
        index=None,
        placeholder="please select a model")
    temperature_input = st.slider("Select temperature", min_value=0.1,step=0.1,value=0.2,max_value=1.0)

st.title("User Story generator")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

def generate_user_story(scenarios):
    '''
      Instantiate LLM model and generate user story
    '''
    llm = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key, temperature=temperature_input)
    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                 "You are an experienced product owner who writes user stories in gherkin format, generate user stories for the scenarios provided in user input. Don't run if user doesnot provide any input"
            ),
            MessagesPlaceholder(variable_name="scenarios")
        ]
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    demo_ephemeral_chat_history = ChatMessageHistory()
    demo_ephemeral_chat_history.add_user_message(scenarios)

    response = chain.invoke({"scenarios": demo_ephemeral_chat_history.messages})
    # Print results
    return st.info(response)


with st.form("myform"):
    scenario_text = st.text_input("Enter scenarios to generate user story:", "")
    submitted = st.form_submit_button("Submit")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    elif not model_name:
        st.info("Please select a model to continue.")
    elif submitted:
        generate_user_story(scenario_text)
