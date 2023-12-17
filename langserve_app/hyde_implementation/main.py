# load the key from .env file
from dotenv import load_dotenv
load_dotenv()

import asyncio
from fastapi import FastAPI



# In[]: Imports

from langchain.llms import OpenAI
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

from langchain.document_loaders import TextLoader
import langchain
# langchain.debug = True

from langchain.chains import RetrievalQA


import sys
sys.path.append("../..")
from do_not_share import CONNECTION_STRING_2 # getting the connection string from to the postgres database
import json
from langserve import add_routes


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)
embeddings = None
llm_hyde = None
llm_query = None
retriever = None
chain = None

def load_model_n_embedding(llm_name: str, embedding_name: str, temperature: float):
    global embeddings
    if embedding_name == "gte":
        embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small") # getting the embedding model with dim 384
    elif embedding_name == "bge_large":
        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en")
    elif embedding_name == "bge_small":
        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en")
    else:
        embeddings = None
        return 
    

    global llm_hyde
    global llm_query

    try:
        if llm_name == "OpenAI":
            llm_hyde = OpenAI(temperature=temperature)
            llm_query = OpenAI(temperature=0)
        else:
            llm_query = HuggingFaceHub(
                    repo_id=llm_name, 
                    model_kwargs={"temperature":0}
                )
            llm_hyde = HuggingFaceHub(
                    repo_id=llm_name, 
                    model_kwargs={"temperature":temperature}
                )
    except:
        llm_query = None
        llm_hyde = None
        return
    global chain
    chain = load_qa_chain( llm=llm_query, chain_type="stuff") #why stuff?
        
    # the model which will be used for generating hypothetical documents
    hyde_embedding_gte = HypotheticalDocumentEmbedder.from_llm(llm_hyde, embeddings, prompt_key="web_search")
    store = LocalFileStore("./cache_gte_pubmed/")
    cached_hyde_embedding_gte = CacheBackedEmbeddings.from_bytes_store(
        hyde_embedding_gte, store, 
    )
    db = PGVector(
        connection_string=CONNECTION_STRING_2,
        embedding_function=cached_hyde_embedding_gte,
        collection_name="pubmed",
        distance_strategy=DistanceStrategy.COSINE,
    ) 

    global retriever
    retriever = db.as_retriever(search_kwargs={'k':3})
    # add_routes(app, retriever)

@app.post("/setvalue")
async def setvalue(llm_name: str, embedding_name: str, temperature: float):
    load_model_n_embedding(llm_name, embedding_name, temperature)
    return_value = ""
    if embeddings == None:
        return_value +="Failed to load embedding; "
    if llm_hyde == None or llm_query == None:
        return_value +="Failed to load llm; "
    return return_value + "Success"

# In[]: llm

# llm_hyde = OpenAI(temperature=0.3)
# llm_query = OpenAI(temperature=0)



# In[]: Implementing the HyDe on embedding

# embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small") # getting the embedding model with dim 384

# # the model which will be used for generating hypothetical documents
# hyde_embedding_gte = HypotheticalDocumentEmbedder.from_llm(llm_hyde, embeddings, prompt_key="web_search")

# store = LocalFileStore("./cache_gte_pubmed/")
# cached_hyde_embedding_gte = CacheBackedEmbeddings.from_bytes_store(
#     hyde_embedding_gte, store, 
# )

# In[]: Loading the db vector with the gte-embedding

# db = PGVector(
# 	connection_string=CONNECTION_STRING_2,
# 	embedding_function=cached_hyde_embedding_gte,
# 	collection_name="pubmed",
# 	distance_strategy=DistanceStrategy.COSINE,
# ) 

# # print("Similarity search for covid-19")
# # print(db.similarity_search("covid-19", k=2))


# retriever = db.as_retriever(search_kwargs={'k':3})
# print("retriever.search_type", retreiver.search_type)

#########################################################################
# query = "What are symptoms of alzheimer's disease?"
# docs = retriever.get_relevant_documents(query)
# reordered_docs = reordering.transform_documents(docs)


# retriever_gte_chain = RetrievalQA.from_llm(llm=llm_query, retriever=retriever, return_source_documents=True)



# chain = load_qa_chain( llm=llm_query, chain_type="stuff") #why stuff? 
# # chain = prompt | model | ...
# answer = chain.run(input_documents = reordered_docs, question=query)
# answer



# In[]: Langserve implementation
# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
# from langserve import add_routes

# add_routes(app, retriever)

from fastapi import Response

# @app.post("/search")
# async def search(query: str):
#     docs = retriever.get_relevant_documents(query)
#     reordered_docs = reordering.transform_documents(docs)
#     return Response(content=reordered_docs, media_type="application/json")
#     # return in json format as required by the client

@app.post("/search")
async def search(query: str):
    docs = retriever.get_relevant_documents(query)
    reordered_docs = reordering.transform_documents(docs)
    global chain
    # chain = prompt | model | ...
    answer = chain.run(input_documents = reordered_docs, question=query)
    
    return {"answer": answer,
            "documents": reordered_docs}
    # return in json format as required by the client
# # query = {"input": {"question": "what do you know about alzheimer", "chat_history": [("hi", "I want to know about medical diseases")]}}

# # print(retreiver.get_relevant_documents(query))
# from fastapi.exceptions import RequestValidationError
# from fastapi.encoders import jsonable_encoder
# from fastapi.responses import JSONResponse
# from fastapi.requests import Request
# from fastapi import status
# from fastapi.responses import StreamingResponse




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
