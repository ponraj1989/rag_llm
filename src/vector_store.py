from langchain.vectorstores import FAISS
from typing import List

class VectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = None

    def create_store(self, documents: List):
        self.store = FAISS.from_documents(documents, self.embeddings)
        return self.store

    def save_store(self, path: str):
        self.store.save_local(path)

    def load_store(self, path: str):
        self.store = FAISS.load_local(path, self.embeddings)
        return self.store

    def get_retriever(self):
        return self.store.as_retriever()