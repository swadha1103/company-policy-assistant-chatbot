from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(folder_path="docs"):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} document pages.")
    print(docs[0].page_content[:500])
