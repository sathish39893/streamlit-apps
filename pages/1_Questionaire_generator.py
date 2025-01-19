''' 
This is an example of generating questions from a pdf file uploaded by the user.
The user can select the model to generate questions from the pdf file.
The model will generate multiple choice questions from the pdf file.
The user can select an answer to the question and the model will validate the answer.
'''
import os
import time
from io import BytesIO
from typing import Optional
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
import streamlit as st
import fitz
from typing_extensions import Annotated, TypedDict

# load .env file
load_dotenv()

# streamlit app
st.set_page_config(
    page_title="Document Questionnaire generator",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="auto",
)

with st.sidebar:
    MODEL_OPTIONS = ["llama-3.1-70b-versatile", "llama3-8b-8192","gemma2-9b-it"]
    groq_api_key = st.text_input(
        "Groq API Key", key="langchain_search_api_key_groq", type="password"
    )
    "[Get an Groq API key](https://console.groq.com/keys)"
    model_name = st.selectbox("Select model",
                          placeholder="choose a model",
                          options=MODEL_OPTIONS)
    NUM_QUESTIONS = st.slider("Select no. of questions", min_value=2,step=1,value=5,max_value=10)


st.title('Document Questionnaire generator')
st.caption("🚀 A Streamlit chatbot powered by Groq")

st.write('Generate multiple choice questionnaire from a pdf file uploaded')

if not groq_api_key or not model_name:
    st.info('Please select a model and enter the Groq API key to continue')
    st.stop()
elif not groq_api_key.startswith('gsk_'):
    st.error('Invalid Groq API key, please enter a valid Groq API key')
    st.stop()
else:
    os.environ["GROQ_API_KEY"] = groq_api_key

file_bytes = st.file_uploader("Upload a PDF file", type=["pdf"])



# Initialization
if 'questions' not in st.session_state:
    st.session_state['questions'] = []

if file_bytes is None:
    st.stop()

# Create a BytesIO object from the byte data
file_stream = BytesIO(file_bytes.getvalue())

# Read the PDF
pages = []
pdf_document = fitz.open(stream=file_stream, filetype="pdf")
for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    pages.append(page)

TEXT_CONTENT = " ".join(d.get_text() for d in pages)

if not TEXT_CONTENT:
    st.write('No text content found in the document')
    st.stop()

st.write(f'Text length found in the document: {len(TEXT_CONTENT)}')

if len(TEXT_CONTENT) > 10000:
    st.write('Text content is too long, please upload a shorter document')
    st.stop()

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

# system prompt to generate questions
SYSTEM_PROMT =  """Generate {questions_count} multiple choice questions from the following text:
'''
{context}
'''
"""

SYSTEM_PROMT_FRMTD = SYSTEM_PROMT.format(context=TEXT_CONTENT, questions_count=NUM_QUESTIONS)

# Create a model
llm = ChatGroq(model=model_name, temperature=0, max_retries=2, rate_limiter=rate_limiter)

# TypedDict
class MCQuestion(TypedDict):
    """Multiple choice question."""
    text: Annotated[str, ..., "The text of the question"]
    options: Annotated[list[str], ..., "The options for the question"]
    answer: Annotated[Optional[int], None, "the answer for the question from the options"]

class MultipleQuestion(TypedDict):
    """List of multiple choice questions."""
    questions: Annotated[list[MCQuestion], ..., "List of multiple choice questions"]


# Cache the data so that it wont trigger the model to generate questions again
@st.cache_data
def call_llm(content: str) -> dict:
    """Call the model to generate questions."""
    # Generate questions
    structured_llm = llm.with_structured_output(MultipleQuestion)
    return structured_llm.invoke([SystemMessage(content=content)])


response = call_llm(SYSTEM_PROMT_FRMTD)
if response is None:
    with st.spinner('loading...'):
        time.sleep(5)

st.text(f'Generated {NUM_QUESTIONS} questions and answers from the document.')
st.text('Please choose an answer to show next question')

st.session_state.questions = response.get("questions")

for question in st.session_state.questions:
    answer_selected = st.radio(
        options=question.get("options"),
        label=question.get("text"),
        index=None)

    if not answer_selected:
        st.text('Please select an answer')
        st.stop()
    else:
        if question.get('options')[question.get('answer')] == answer_selected:
            st.text(f"✅ Correct! The answer is {answer_selected}")
        else:
            st.text('❌ Incorrect answer, please try again')
            st.stop()

st.balloons()
st.success('Congratulations! You have completed the questionnaire', icon="✔")
