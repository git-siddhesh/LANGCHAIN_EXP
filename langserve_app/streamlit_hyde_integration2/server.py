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


import sys
sys.path.append("../..")
from do_not_share import CONNECTION_STRING_2 # getting the connection string from to the postgres database
import json
from langserve import add_routes
DOC_SPACE_DIR_ = './faiss_doc_space'


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
r_chain = None
def load_model_n_embedding_hyde(llm_name: str, embedding_name: str, temperature: float, query_temperature: float):
    global embeddings
    if embedding_name == "thenlper/gte-large":
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

    # try:
    if llm_name == "OpenAI":
        llm_hyde = OpenAI(temperature=temperature)
        llm_query = ChatOpenAI(model="gpt-3.5-turbo", temperature=query_temperature)
    else:
        llm_query = HuggingFaceHub(
                repo_id=llm_name, 
                model_kwargs={"temperature":query_temperature}
            )
        llm_hyde = HuggingFaceHub(
                repo_id=llm_name, 
                model_kwargs={"temperature":temperature}
            )
    # except:
    #     llm_query = None
    #     llm_hyde = None
    #     return
    global chain
    chain = load_qa_chain( llm=llm_query, chain_type="stuff") #why stuff?
        
    # the model which will be used for generating hypothetical documents
    hyde_embedding_gte = HypotheticalDocumentEmbedder.from_llm(llm_hyde, embeddings, prompt_key="web_search")
    print("HyDe Embedding created")
    store = LocalFileStore("./cache_gte_pubmed/")
    cached_hyde_embedding_gte = CacheBackedEmbeddings.from_bytes_store(
        hyde_embedding_gte, store, 
    )
    global document_search_space
    db = PGVector(
        connection_string=CONNECTION_STRING_2,
        embedding_function=cached_hyde_embedding_gte,
        collection_name="pubmed",
        distance_strategy=DistanceStrategy.COSINE,
    ) 
    document_search_space = db
    print("DB created/Loaded")
    global retriever
    retriever = db.as_retriever(search_kwargs={'k':3})
    # add_routes(app, retriever)
    global r_chain
    r_chain = RetrievalQA.from_llm(llm=llm_query, retriever=retriever, return_source_documents=True)

def load_model_n_embedding(llm_name: str, embedding_name: str, temperature: float):
    print(llm_name, embedding_name, temperature)
    global embeddings
    try:
        if embedding_name == "openai-gpt":
            embeddings = OpenAIEmbeddings() # getting the embedding model with dim 384
        else:
            embeddings = HuggingFaceEmbeddings(model_name=embedding_name)
    except Exception as e: 
        print("Error in embedding load: ",e)
        embeddings = None
        return   

    global llm_query
    global llm_hyde

    # try:
    if llm_name == "openai-gpt":
        llm_hyde = OpenAI(temperature=0.3)
        llm_query= ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
    else:
        llm_query = HuggingFaceHub(
                repo_id=llm_name, 
                model_kwargs={"temperature":temperature}
            )
        llm_hyde = HuggingFaceHub(
                repo_id=llm_name, 
                model_kwargs={"temperature":0.3}
            )
        
    # except:
    #     llm_query = None
    #     llm_hyde = None
    #     return
    global chain
    chain = load_qa_chain( llm=llm_query, chain_type="stuff") #why stuff?
    hyde_embedding_gte = HypotheticalDocumentEmbedder.from_llm(llm_hyde, embeddings, prompt_key="web_search")
    print("HyDe Embedding created")
    embeddings = hyde_embedding_gte


@app.post("/setvalue")
async def setvalue(llm_name: str, embedding_name: str, temperature: float, DOC_SPACE_DIR: str):
    global flag
    return_value = ""
    if flag:
        global DOC_SPACE_DIR_
        DOC_SPACE_DIR_ = DOC_SPACE_DIR
        load_model_n_embedding(llm_name, embedding_name, temperature)
        flag = False
        if embeddings == None:
            return_value +="Failed to load embedding; "
        if llm_query == None:
            return_value +="Failed to load llm; "
        return return_value + "Success"
    else:
        return return_value + "Already set"


flag = True
@app.post("/setvaluehyde")
async def setvaluehyde(llm_name: str, embedding_name: str, temperature: float, query_temperature: float):
    global flag
    return_value = ""
    if flag:
        load_model_n_embedding_hyde(llm_name, embedding_name, temperature, query_temperature)
        flag = False
        if embeddings == None:
            return_value +="Failed to load embedding; "
        if llm_hyde == None or llm_query == None:
            return_value +="Failed to load llm; "
        return return_value + "Success"
    else:
        return return_value + "Already set"


def get_text(doc_path = '/home/dosisiddhesh/LANGCHAIN_EXP/pdfs', uploaded_file = False, chunk_size = 500, chunk_overlap = 100):
    myPdfReader = None
    raw_text = ''
    if uploaded_file == False:
        pdf_files = [os.path.join(doc_path, f) for f in os.listdir(doc_path) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            myPdfReader = PdfReader(pdf_file)
            for page in myPdfReader.pages:
                raw_text += page.extract_text()
    # else:
    #     myPdfReader = PdfReader(doc_path)
    #     for page in myPdfReader.pages:
    #         raw_text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # # show the length to the streamlit as log
    # with st.empty():
    #     st.info(f"Length of the chunks: {len(texts)}")
    #     # time.sleep(5)
    #     st.info("Creating the vector store")
    return texts    

from langchain.vectorstores import FAISS
document_search_space = None
flag2 = True
@app.post("/start")
async def start(texts:str, chunk_size:int, chunk_overlap:int):
    global flag2
    global embeddings
    global retriever
    global document_search_space
    global r_chain
    global llm_query

    texts = get_text(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # print("Texts: ",texts)
    # print("type: ",type(texts))
    # input("Press Enter to continue...")
    return_value = ""
    if flag2:
        if not os.path.exists(DOC_SPACE_DIR_):
            document_search_space = FAISS.from_texts(texts, embeddings)
            document_search_space.save_local(DOC_SPACE_DIR_)
        else:
            document_search_space = FAISS.load_local(DOC_SPACE_DIR_, embeddings)

        retriever = document_search_space.as_retriever(search_kwargs={'k':3})
        r_chain = RetrievalQA.from_llm(llm=llm_query, retriever=retriever, return_source_documents=True)


        flag2 = False
        return return_value + "Success"
    else:
        return return_value + "Already set"
    
@app.post("/isdbexists")
async def isdbexists():
    global DOC_SPACE_DIR_
    global document_search_space
    global retriever
    global embeddings

    print(DOC_SPACE_DIR_)
    if os.path.exists(DOC_SPACE_DIR_):
        document_search_space = FAISS.load_local(DOC_SPACE_DIR_, embeddings)
        retriever = document_search_space.as_retriever(search_kwargs={'k':3})

        return True
    return False

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import Query
from typing import List
from operator import itemgetter

@app.post("/search")
async def search(query: str, history_qna: List[str] = Query(...)):

    global document_search_space
    # docs = document_search_space.similarity_search(query,k = 5)
    #------------------------------------------------------------------
    reordered_docs = reordering.transform_documents(retriever.get_relevant_documents(query))
    global chain
    global llm_query
    # prompt = PromptTemplate(
    #     template="Answer the following question: {question}",
    #     question=query,
    # )

    template = """Follow the Previous chat history: {history_qna}
    Answer the question based only on the following context:
    {context}
    The Question is: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {   "history_qna": itemgetter("history_qna"),
        # {   "history_qna": RunnablePassthrough,
            "context": retriever | format_docs, 
            # "question": RunnablePassthrough()
            "question": itemgetter("question")
        }
        | prompt
        | llm_query
        | output_parser
    )
    # chain = prompt | llm_query | output_parser

    # print(reordered_docs)
    # print(history_qna)
    # query = "; ".join(history_qna) + "; " + query
    print(query)
    answer = chain.invoke({"history_qna": "; ".join(history_qna), "question": query})
    #------------------------------------------------------------------
    # answer = chain.run(input_documents = reordered_docs, question=query)
    #------------------------------------------------------------------
    # global r_chain
    # result = await r_chain.ainvoke( "what is covid-19?")
    # answer = result["result"]
    # reordered_docs = result["source_documents"]
    print(answer)
    return {"answer": answer,
            "documents": reordered_docs}

@app.post("restart")
async def restart():
    global flag
    global flag2
    flag = True
    flag2 = True
    return 'Success'

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)

# %%
