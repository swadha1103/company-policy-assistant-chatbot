from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from document_loader import load_documents
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()

def create_vector_store():
    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local("faiss_index")
    print("Vector store saved locally as 'faiss_index' folder.")


if __name__ == "__main__":
    create_vector_store()
