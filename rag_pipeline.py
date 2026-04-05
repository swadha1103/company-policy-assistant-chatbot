import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# ---------------- LOAD PDFs ----------------
pdf_files = [
    "docs/employee_handbook.pdf",
    "docs/leave_policy.pdf",
    "docs/wfh_policy.pdf"
]

documents = []
for file in pdf_files:
    loader = PyPDFLoader(file)
    docs_loaded = loader.load()
    for d in docs_loaded:
        d.metadata["source"] = file
    documents.extend(docs_loaded)

print(f"Loaded {len(documents)} pages.")

# ---------------- SPLIT INTO CHUNKS ----------------
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
docs = splitter.split_documents(documents)

# ---------------- CLEAN TEXT ----------------
cleaned_docs = []
for d in docs:
    text = d.page_content.replace("\n", " ").strip()
    text = text.replace("ABC Tech Solutions Pvt. Ltd.", "")
    text = re.sub(r"\s+", " ", text)
    if len(text) > 50:
        d.page_content = text
        cleaned_docs.append(d)

docs = cleaned_docs
print(f"{len(docs)} clean chunks ready.")

# ---------------- EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build vector store for retrieval
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# ---------------- SEMANTIC SENTENCE MATCHER ----------------
def extract_best_sentence(question, retrieved_docs):
    sentences = []
    for doc in retrieved_docs:
        for s in re.split(r'(?<=[.!?]) +', doc.page_content):
            s = s.strip()
            if 30 < len(s) < 300:
                sentences.append(s)

    if not sentences:
        return "I don't know"

    # Embed question and sentences
    q_emb = embeddings.embed_query(question)
    s_embs = embeddings.embed_documents(sentences)

    # Cosine similarity
    scores = [np.dot(q_emb, s_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(s_emb)) for s_emb in s_embs]

    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    # Confidence threshold prevents wrong answers
    if best_score < 0.35:
        return "I don't know"

    return sentences[best_idx]


# ---------------- CHAT LOOP ----------------
if __name__ == "__main__":
    while True:
        q = input("\nYou: ")
        if q.lower() == "exit":
            print("👋 Goodbye!")
            break

        retrieved_docs = retriever.invoke(q)
        answer = extract_best_sentence(q, retrieved_docs)

        print("\nBot:", answer)
