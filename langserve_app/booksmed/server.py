# load the key from .env file
from dotenv import load_dotenv
load_dotenv()

import asyncio
from fastapi import FastAPI

# In[]: Imports

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceBgeEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.document_transformers import LongContextReorder
reordering = LongContextReorder()
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import StrOutputParser
output_parser = StrOutputParser()

from langchain.document_loaders import TextLoader
import langchain
import os
# langchain.debug = True

from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import Query
from typing import List
from operator import itemgetter


import sys
sys.path.append("../..")
from do_not_share import CONNECTION_STRING_2 # getting the connection string from to the postgres database
import json
from langserve import add_routes
DOC_SPACE_DIR_ = './faiss_doc_space'

########################################################################################################################
# Global Variables
llm_query = None
llm_hyde = None
embeddings = None
chain = None
db = None



# #-----------------------------------------------------------------------------------------------------------------------
# app = FastAPI(
#     title="LangChain Server",
#     version="1.0",
#     description="A simple API server using LangChain's Runnable interfaces",
# )

def load_model_n_embedding(hyde_llm_name: str, query_llm_name: str, embedding_name: str, temp_hyde: float, temp_query: float):
    global llm_query
    global llm_hyde
    global embeddings
    global chain
    global retriever
    global document_search_space
    try:
        if embedding_name == "openai-gpt":
            embeddings = OpenAIEmbeddings() # getting the embedding model with dim 384
        else:
            embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
    except Exception as e: 
        print("Error in embedding load: ",e)
        embeddings = None
        return   

    # try:
    if hyde_llm_name == "openai-gpt":
        llm_hyde = OpenAI(temperature=temp_hyde)
    else:
        llm_hyde = HuggingFaceHub(repo_id=hyde_llm_name, model_kwargs={"temperature":temp_hyde})
    if query_llm_name == "openai-gpt":
        llm_query= ChatOpenAI(model="gpt-3.5-turbo", temperature=temp_query)
    else:
        llm_query = HuggingFaceHub(
                repo_id=query_llm_name, 
                model_kwargs={"temperature":temp_query}
            )        
    # chain = load_qa_chain( llm=llm_query, chain_type="stuff") #why stuff?
    hyde_embedding_gte = HypotheticalDocumentEmbedder.from_llm(llm_hyde, embeddings, prompt_key="web_search")
    print("HyDe Embedding created")

    embeddings = hyde_embedding_gte

    store = LocalFileStore("./cache_gte_pubmed/")
    cached_hyde_embedding_gte = CacheBackedEmbeddings.from_bytes_store(
        hyde_embedding_gte, store, 
    )
    try:
        db = PGVector(
            connection_string=CONNECTION_STRING_2,
            embedding_function=cached_hyde_embedding_gte,
            collection_name="pubmed",
            distance_strategy=DistanceStrategy.COSINE,
        ) 
        document_search_space = db
    except Exception as e:
        print("Error in db creation: ",e)
        
    print("DB created/Loaded")
    retriever = document_search_space.as_retriever(search_kwargs={'k':3})


load_model_n_embedding("openai-gpt", "openai-gpt", "openai-gpt", 0.9, 0.9)

# #-----------------------------------------------------------------------------------------------------------------------

query = "What is the best treatment for COVID-19?"
history_qna = []

# template = """Follow the Previous chat history: {history_qna}
# Answer the question based only on the following context:
# {context}
# The Question is: {question}
# """
template = """
Answer the question based only on the following context:
{context}
The Question is: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
reordered_docs = None
def format_docs(docs):
    global reordered_docs
    reordered_docs = reordering.transform_documents(docs)
    return "\n\n".join([d.page_content for d in reordered_docs])

chain = (
    {   
        # "history_qna": itemgetter("history_qna"),
        "context": retriever | format_docs, 
        # "question": RunnablePassthrough()
        "question": itemgetter("question")
    }
    | prompt
    | llm_query
    | output_parser
)

answer = chain.invoke({
    # "history_qna": "; ".join(history_qna), 
    "question": query})
#------------------------------------------------------------------
print(query)
print("docs",answer)
print("docs: ",retriever.search(query))
# return {"answer": answer,
#         "documents": reordered_docs}