import streamlit as st
import re
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Company Policy Bot", page_icon="📄")
st.title("📄 Company Policy Assistant")
st.write("Ask questions about company policies (leave, WFH, security, etc.)")

# ---------------- LOAD SYSTEM ONCE ----------------
@st.cache_resource
def load_rag_system():
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    cleaned_docs = []
    for d in docs:
        text = d.page_content.replace("\n", " ").strip()
        text = text.replace("ABC Tech Solutions Pvt. Ltd.", "")
        text = re.sub(r"\s+", " ", text)
        if len(text) > 50:
            d.page_content = text
            cleaned_docs.append(d)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(cleaned_docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    return retriever, embeddings

retriever, embeddings = load_rag_system()

# ---------------- SEMANTIC ANSWER FUNCTION ----------------
def extract_best_sentence(question, retrieved_docs):
    sentences = []
    for doc in retrieved_docs:
        for s in re.split(r'(?<=[.!?]) +', doc.page_content):
            s = s.strip()
            if 30 < len(s) < 300:
                sentences.append(s)

    if not sentences:
        return "I don't know"

    q_emb = embeddings.embed_query(question)
    s_embs = embeddings.embed_documents(sentences)

    scores = [np.dot(q_emb, s_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(s_emb)) for s_emb in s_embs]

    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    if best_score < 0.35:
        return "I don't know"

    return sentences[best_idx]

# ---------------- SESSION STATE ----------------
if "answer" not in st.session_state:
    st.session_state.answer = None

if "show_answer" not in st.session_state:
    st.session_state.show_answer = False


# ---------------- UI ----------------
st.markdown("💡 Type your question and click **Submit**. To stop, simply close the browser tab.")

with st.form("question_form"):
    question = st.text_input("Enter your question:")
    submit = st.form_submit_button("Submit")

# When user submits
if submit and question.strip() != "":
    with st.spinner("Searching company policies..."):
        retrieved_docs = retriever.invoke(question)
        answer = extract_best_sentence(question, retrieved_docs)

    st.session_state.answer = answer
    st.session_state.show_answer = True

# If user is typing (form not submitted), hide answer
if not submit:
    st.session_state.show_answer = False

# Show answer only if flag is True
if st.session_state.show_answer and st.session_state.answer:
    st.markdown("### 🤖 Answer")
    st.success(st.session_state.answer)
