from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

def load_documents(path="docs"):
    if not os.path.exists(path):
        FileNotFoundError(f"Path {path} does not exist")

    loader = DirectoryLoader(
        path=path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()
    return documents


def split_documents(documents,chunk_size=1000,chunk_overlap=0):
    text_spliter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_spliter.split_documents(documents)
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vactor_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return vactor_store


documents = load_documents("docs")

# Step 2: Split into chunks
chunks = split_documents(documents, chunk_size=500)

# # Step 3: Create vector store
vectorstore = create_vector_store(chunks)