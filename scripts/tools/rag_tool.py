from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

import os

from typing import List

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_chroma import Chroma

 

from langchain.tools import tool

from pydantic import BaseModel
 

def load_documents(folder_path: str) -> List[Document]:
    ## Load documents
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents

 

folder_path = "/home/m1cgw01/adap/windows.groups/statdata/Projects/CALLME/comment_documents/R-1523"
documents = load_documents(folder_path)
print(f"Loaded {len(documents)} documents from the folder.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

splits = text_splitter.split_documents(documents)


embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

collection_name = "my_collection"
vectorstore = Chroma.from_documents(
    collection_name=collection_name,
    documents=splits,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)

class RagToolSchema(BaseModel):
    question: str

@tool(args_schema=RagToolSchema)
def retriever_tool(question):
    """Tool to Retrieve Semantically Similar documents to answer User Questions related to FutureSmart AI"""
    print("INSIDE RETRIEVER NODE")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retriever_results = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in retriever_results)