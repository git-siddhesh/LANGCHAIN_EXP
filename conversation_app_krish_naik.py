import os
# from openai_key import HUGGING_FACE_KEY
# from openai_key import OPENAI_API_KEY
# # os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
# from langchain.llms import HuggingFaceHub
from sentence_transformers.util import cos_sim

from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI


import streamlit as st
st.set_page_config(
    page_title= "MedBuddy", 
    layout="centered", 
    initial_sidebar_state="collapsed",
    menu_items={
        'Get help': None,
        'About': None,
    })
st.header("Hey, I'm MedBuddy!, your personal medical assistant.")

#_______________________________________________________________________________________
#--------- load the model------------------------------------------------
# llm = HuggingFaceHub(
#     repo_id='mistralai/Mistral-7B-Instruct-v0.1', 
#     model_kwargs={"temperature":0.5}
#     )

llm = ChatOpenAI( temperature=0.5 )

#_______________________________________________________________________________________
#---Initializatio: set the initial system message -> flowMessage------------------------
if 'flowMessage' not in st.session_state:
    st.session_state['flowMessage'] = [
        SystemMessage(content="Hi, I'm MedBuddy, your personal medical assistant. How can I help you?")
    ]
#_______________________________________________________________________________________
# flowMessage = [ HumanMessage(question) + AlMessage(answer) ]
def get_chat_model_response(question):
    st.session_state['flowMessage'].append(HumanMessage(content=question))
    answer = llm(st.session_state['flowMessage'])
    st.session_state['flowMessage'].append(AIMessage(content=answer.content))
#_______________________________________________________________________________________
#---- display the conversation history---------------------------------------------------

if input_text := st.chat_input("Ask me anything about your health", key='input'):
    get_chat_model_response(input_text)
    for message in st.session_state['flowMessage']:
        with st.chat_message(name=message.type):
            st.markdown(message.content)
#_______________________________________________________________________________________