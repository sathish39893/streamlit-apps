import json
import re
import ast
import datetime
import random
from typing import Callable

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    json_text = match.group(0)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(json_text)
        except Exception:
            return {}


def safe_calculator(expression: str) -> str:
    cleaned = re.sub(r"[^0-9\.\+\-\*/\(\) \\n]", "", expression)
    try:
        result = eval(cleaned, {"__builtins__": None}, {})
        return str(result)
    except Exception as exc:
        return f"Calculator error: {exc}"


def current_time(_: str) -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def reverse_text(text: str) -> str:
    return text[::-1]


def random_number(range_spec: str) -> str:
    parts = re.findall(r"-?\d+", range_spec)
    if len(parts) >= 2:
        a, b = int(parts[0]), int(parts[1])
        if a > b:
            a, b = b, a
        return str(random.randint(a, b))
    return "Invalid range: please provide two integers."


def text_stats(text: str) -> str:
    words = re.findall(r"\w+", text)
    sentences = re.findall(r"[.!?]", text)
    return json.dumps(
        {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "char_count": len(text),
        }
    )


TOOL_INSTRUCTIONS = (
    "Available tools:\n"
    "- calculator: Evaluate arithmetic expressions (e.g. 7 * 6 + 3).\n"
    "- current_time: Return the current local date and time.\n"
    "- text_stats: Count words, sentences, and characters in a text string.\n"
    "When no tool is needed, return tool_name as 'none'.\n"
)


def run_deep_agent(question: str, llm: ChatOpenAI) -> dict:
    plan_prompt = [
        SystemMessage(
            content=(
                "You are a deep reasoning agent. Read the user question, decide whether a tool is needed, "
                "and respond with valid JSON containing plan, tool_name, tool_input, and final_answer. "
                "Use tool_name 'none' when no tool is required."
            )
        ),
        HumanMessage(
            content=(
                f"Question: {question}\n\n{TOOL_INSTRUCTIONS}\n\n"
                "Return only the JSON object with these keys:\n"
                "{\n"
                "  \"plan\": \"...\",\n"
                "  \"tool_name\": \"calculator\" or \"current_time\" or \"text_stats\" or \"reverse_text\" or \"random_number\" or \"none\",\n"
                "  \"tool_input\": \"...\",\n"
                "  \"final_answer\": \"...\"\n"
                "}\n"
            )
        ),
    ]
    response = llm.invoke(plan_prompt)
    text = getattr(response, "content", str(response))
    plan_data = extract_json(text)

    if plan_data.get("tool_name") and plan_data.get("tool_name") != "none":
        tool_name = plan_data.get("tool_name")
        tool_input = plan_data.get("tool_input", "")
        tool_map: dict[str, Callable[[str], str]] = {
            "calculator": safe_calculator,
            "current_time": current_time,
            "text_stats": text_stats,
            "reverse_text": reverse_text,
            "random_number": random_number,
        }
        tool_result = tool_map.get(tool_name, lambda _: f"Unknown tool: {tool_name}")(tool_input)
        final_prompt = [
            SystemMessage(
                content=(
                    "You are a deep reasoning agent. Use the tool observation below to produce a final answer. "
                    "Reply with a concise answer only."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {question}\n"
                    f"Plan: {plan_data.get('plan', '')}\n"
                    f"Tool: {tool_name}\n"
                    f"Tool input: {tool_input}\n"
                    f"Observation: {tool_result}\n"
                )
            ),
        ]
        final_response = llm.invoke(final_prompt)
        final_answer = getattr(final_response, "content", str(final_response))
        plan_data["tool_result"] = tool_result
        plan_data["final_answer"] = final_answer.strip()

    return plan_data


st.set_page_config(
    page_title="Chainlit Deep Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.header("Deep Agent settings")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        key="deep_agent_openai_api_key",
        type="password",
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    model_name = st.selectbox(
        "Select a model",
        options=(
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
        ),
        index=0,
    )
    temperature_input = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
    )

st.title("Simple Deep Agent")
st.caption("A minimal OpenAI-powered deep agent flow with planning, tool execution, and final answer generation.")

st.markdown(
    "This page demonstrates a small deep agent pattern: first the model decides whether a tool is needed, "
    "then the tool is executed if required, and finally the agent writes the final answer.\n\n"
    "Available tools:\n"
    "- calculator: Evaluate arithmetic expressions.\n"
    "- current_time: Return the current local date and time.\n"
    "- text_stats: Count words, sentences, and characters.\n"
    "- reverse_text: Reverse the provided text.\n"
    "- random_number: Generate a random integer from a range."
)

question = st.text_area(
    "Ask the agent a question:",
    value="What is 27 * 19?",
    height=150,
)

if not openai_api_key:
    st.warning("Enter your OpenAI API key in the sidebar to run the agent.")
    st.stop()

llm = ChatOpenAI(model=model_name, api_key=openai_api_key, temperature=temperature_input)

if st.button("Run Deep Agent"):
    with st.spinner("Thinking like a deep agent..."):
        result = run_deep_agent(question, llm)

    if not result:
        st.error("Unable to parse the model response. Try simplifying your question.")
    else:
        st.subheader("Agent plan")
        st.write(result.get("plan", "No plan was generated."))

        st.subheader("Tool execution")
        tool_name = result.get("tool_name", "none")
        if tool_name and tool_name != "none":
            st.write(f"**Tool:** {tool_name}")
            st.write(f"**Tool input:** {result.get('tool_input', '')}")
            st.write(f"**Tool result:** {result.get('tool_result', '')}")
        else:
            st.write("No tool was needed for this question.")

        st.subheader("Final answer")
        st.info(result.get("final_answer", "No final answer generated."))
