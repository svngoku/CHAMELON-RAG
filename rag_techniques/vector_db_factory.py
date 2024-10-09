from langchain_community.vectorstores import FAISS, Qdrant, Pinecone, VectorStore
from langchain_chroma import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from rag_techniques.loaders import FileLoader
import os
from typing import List, Union

class VectorDBFactory:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings()

    def create_vectorstore(self, data: Union[List[str], List[Document], List[str]], store_type: str = "faiss") -> VectorStore:
        # Define supported vector store classes
        store_classes = {
            "faiss": FAISS,
            "qdrant": Qdrant,
            "pinecone": Pinecone,
            "chromadb": ChromaDB
        }

        # Load documents based on input type
        documents = self._load_documents(data)

        # Select the appropriate vector store class
        if store_type.lower() == "chromadb":
            return Chroma(
                collection_name="example_collection",
                embedding_function=self.embeddings,
                persist_directory="./chroma_langchain_db"
            )
        else:
            store_class = store_classes.get(store_type.lower(), FAISS)
            return store_class.from_documents(documents, self.embeddings)

    def _load_documents(self, data: Union[List[str], List[Document]]) -> List[Document]:
        """Load documents from file paths or prepare them from strings."""
        if isinstance(data[0], str) and os.path.isfile(data[0]):
            loader = FileLoader()
            return loader.load_files(data)
        else:
            return self._prepare_documents(data)

    def _prepare_documents(self, data: Union[List[str], List[Document]]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        if isinstance(data[0], str):
            texts = []
            for text in data:
                texts.extend(text_splitter.split_text(text))
            return [Document(page_content=t) for t in texts]
        elif isinstance(data[0], Document):
            return text_splitter.split_documents(data)
        else:
            raise ValueError("Input must be a list of strings or a list of Document objects")
