from langchain.embeddings import HuggingFaceEmbeddings

class EmbeddingManager:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def get_embeddings(self):
        return self.embeddings