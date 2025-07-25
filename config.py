import os

# Model configurations
LLM_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)