from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from typing import List
from rag_techniques.base import BaseRetriever
from langchain_community.vectorstores import VectorStore
from pydantic import Field

class SimpleRetriever(BaseRetriever):
    vectorstore: VectorStore = Field(description="Vector store for document retrieval")

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for the given query."""
        return self.vectorstore.similarity_search(query, k=k)

    # Keep the retrieve method for backward compatibility
    def retrieve(self, query: str, k: int = 5) -> List[Document]:   
        """Alias for get_relevant_documents."""
        return self.get_relevant_documents(query, k)
    
