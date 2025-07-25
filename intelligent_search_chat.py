import os
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredFileLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Supported file extensions
SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.json', '.html', '.jpg', '.png', '.csv', '.xlsx']

def ingest_files(data_dir: str) -> List:
    docs = []
    for fname in os.listdir(data_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            loader = UnstructuredFileLoader(os.path.join(data_dir, fname))
            docs.extend(loader.load())
    return docs

def build_vector_store(docs: List) -> FAISS:
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def create_local_llm(model_name="meta-llama/Llama-2-7b-chat-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

def create_chat_chain(vector_store: FAISS):
    llm = create_local_llm()
    chain = ConversationalRetrievalChain.from_llm(llm, vector_store.as_retriever())
    return chain

def main():
    data_dir = "./data"
    print("Ingesting files...")
    docs = ingest_files(data_dir)
    print(f"Ingested {len(docs)} documents.")
    print("Building vector store...")
    vector_store = build_vector_store(docs)
    print("Creating chat chain...")
    chat_chain = create_chat_chain(vector_store)

    print("Ready for Q&A. Type 'exit' to quit.")
    chat_history = []
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break
        result = chat_chain({"question": query, "chat_history": chat_history})
        print("Bot:", result["answer"])
        chat_history.append((query, result["answer"]))

if __name__ == "__main__":
    main()