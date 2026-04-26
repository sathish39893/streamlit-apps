"""
Simple Chat Interface with OpenAI Agents SDK
This page demonstrates a basic chat interface using OpenAI's Agents SDK with guardrails and tools.
"""

import streamlit as st
from agents import Agent, Runner, InputGuardrailResult, function_tool, input_guardrail
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="OpenAI Agents Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("OpenAI Agents Chat")
st.caption("🚀 A simple chat interface powered by OpenAI Agents SDK with guardrails and tools")

# Sidebar for configuration
with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        key="openai_api_key", 
        type="password"
    )
    "[Get an OpenAI API key](https://platform.openai.com/api-keys)"
    
    model_name = st.selectbox(
        "Select model",
        options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        index=1
    )

# Check if API key is provided
if not openai_api_key:
    st.info("Please enter your OpenAI API key to continue")
    st.stop()

# Set API key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Define a tool
@function_tool
def get_current_time():
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Define input guardrail
@input_guardrail
def input_guardrail_func(ctx, agent, input_data):
    """Simple guardrail that blocks messages containing inappropriate words."""
    inappropriate_words = ["bad", "inappropriate", "offensive"]  # Example words
    message = input_data.get("messages", [])[-1].get("content", "").lower()
    
    for word in inappropriate_words:
        if word in message:
            return InputGuardrailResult(
                triggered=True, 
                message=f"Message blocked: contains inappropriate content ('{word}')"
            )
    
    return InputGuardrailResult(triggered=False)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create agent
@st.cache_resource
def get_agent(model_name):
    return Agent(
        name="Assistant",
        instructions="You are a helpful AI assistant. You can use tools to help answer questions. For example, you can get the current time if asked.",
        model=model_name,
        input_guardrails=[input_guardrail_func],  # Add guardrail
        tools=[get_current_time]  # Add tool
    )

agent = get_agent(model_name)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare messages for the agent
                messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]
                
                # Run the agent
                result = Runner.run(agent, messages)
                
                # Get the response
                response = result.final_output
                
                # Display response
                st.markdown(response)
                
                # Add assistant message to session state
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})