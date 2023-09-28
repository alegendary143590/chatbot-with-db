import os
import torch

# from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.docstore.document import Document
from typing import List
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc
    
# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = "prefix_SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = "prefix_DB"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Show sources
SHOW_SOURCES = True

# Device Type
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".csv": CSVLoader,
    # ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".enex": EverNoteLoader,
    ".eml": MyElmLoader,
    ".epub": UnstructuredEPubLoader,
    ".html": UnstructuredHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
    ".odt": UnstructuredODTLoader,
    ".pdf": PyMuPDFLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".txt": TextLoader,
}

# Default Instructor Model
#EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
# You can also choose a smaller model, don't forget to change HuggingFaceInstructEmbeddings
# to HuggingFaceEmbeddings in both ingest.py and run_localGPT.py
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"