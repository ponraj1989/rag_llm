from typing import List
import os
from langchain.document_loaders import (
    UnstructuredFileLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredImageLoader,
    UnstructuredHTMLLoader,
)

class DocumentLoader:
    LOADER_MAPPING = {
        '.pdf': UnstructuredFileLoader,
        '.docx': UnstructuredFileLoader,
        '.json': JSONLoader,
        '.html': UnstructuredHTMLLoader,
        '.jpg': UnstructuredImageLoader,
        '.png': UnstructuredImageLoader,
        '.csv': CSVLoader,
    }

    @staticmethod
    def load_documents(data_dir: str) -> List:
        documents = []
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            ext = os.path.splitext(file_name)[1].lower()
            
            if ext in DocumentLoader.LOADER_MAPPING:
                loader_class = DocumentLoader.LOADER_MAPPING[ext]
                try:
                    loader = loader_class(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        
        return documents