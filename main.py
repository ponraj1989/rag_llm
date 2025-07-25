from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingManager
from src.llm_interface import LLMInterface
from src.vector_store import VectorStore
from langchain.chains import ConversationalRetrievalChain
import config

def initialize_system():
    # Load documents
    print("Loading documents...")
    documents = DocumentLoader.load_documents(config.DATA_DIR)
    print(f"Loaded {len(documents)} documents")

    # Initialize embeddings
    print("Initializing embeddings...")
    embedding_manager = EmbeddingManager(config.EMBEDDING_MODEL_NAME)
    embeddings = embedding_manager.get_embeddings()

    # Create or load vector store
    print("Setting up vector store...")
    vector_store = VectorStore(embeddings)
    try:
        vs = vector_store.load_store(config.VECTOR_STORE_PATH)
        print("Loaded existing vector store")
    except:
        vs = vector_store.create_store(documents)
        vector_store.save_store(config.VECTOR_STORE_PATH)
        print("Created new vector store")

    # Initialize LLM
    print("Initializing LLM...")
    llm_interface = LLMInterface(config.LLM_MODEL_NAME)
    llm = llm_interface.get_llm()

    # Create conversation chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.get_retriever()
    )
    
    return chain

def main():
    chain = initialize_system()
    chat_history = []
    
    print("\nReady for questions! (Type 'exit' to quit)")
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break
            
        result = chain({"question": query, "chat_history": chat_history})
        print("\nAssistant:", result["answer"])
        chat_history.append((query, result["answer"]))

if __name__ == "__main__":
    main()