# In[]: Environment Setup
#______________________________________________________________________________________________________________________
import os
os.environ['OPENAI_API_KEY'] = ''

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory 
from langchain.chains import (LLMChain, SimpleSequentialChain, SequentialChain)
import streamlit as st


llm = OpenAI(temperature=0.8)

# In[]: Streamlit Demo
#______________________________________________________________________________________________________________________
st.title("Music Marshal")
input_text = st.text_input("Your Query")
# output_text = llm(input_text)
# st.write(output_text)



# In[]: Prompt Template
#______________________________________________________________________________________________________________________
uno_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about song {name}."
)

dos_prompt = PromptTemplate(
    input_variables = ['song_detail'],
    template = "Who is the singer of {song_detail}."
)

tercero_prompt = PromptTemplate(
    input_variables = ['singer_name'],
    template = "What are top 5 songs of {singer_name}."
)

# In[]: Memory
#______________________________________________________________________________________________________________________
song_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
singer_memory = ConversationBufferMemory(input_key='song_detail', memory_key='chat_history')
top_5_memory = ConversationBufferMemory(input_key='singer_name', memory_key='top_5_songs_history') 


# In[]: Chain
#______________________________________________________________________________________________________________________
chain_uno = LLMChain(llm=llm, prompt=uno_prompt, verbose=True, output_key='song_detail', memory=song_memory)
chain_dos = LLMChain(llm=llm, prompt=dos_prompt, verbose=True, output_key='singer_name', memory=singer_memory)
chain_tercero = LLMChain(llm=llm, prompt=tercero_prompt, verbose=True, output_key='top_5_songs', memory=top_5_memory)

# main_chain = SimpleSequentialChain(chains=[ llm_chain, chain_dos], verbose=True)
main_chain = SequentialChain(
    chains=[chain_uno, chain_dos, chain_tercero ], input_variables=['name'], output_variables=['song_detail', 'singer_name', 'top_5_songs'], verbose=True)

# In[]: Streamlit Demo Output
#______________________________________________________________________________________________________________________
if input_text:
    # st.write(main_chain.run(input_text)) #ValueError: `run` not supported when there is not exactly one output key. Got ['song_detail', 'singer_name', 'top_5_songs'].
    st.write(main_chain({'name': input_text}))
    with st.expander("Singer Name"):
        st.info(song_memory.buffer)
    with st.expander("Song Detail"):
        st.info(singer_memory.buffer)
    with st.expander("Top 5 Songs"):
        st.info(top_5_memory.buffer)

