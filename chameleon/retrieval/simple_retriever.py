from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from chameleon.base import BaseRetriever, RetrieverConfig
from langchain_community.vectorstores import VectorStore
import logging
from chameleon.utils.logging_utils import COLORS

class SimpleRetriever(BaseRetriever):
    """Simple retriever implementation using vector store similarity search."""
    
    def __init__(self, vectorstore: VectorStore, search_kwargs: Optional[Dict[str, Any]] = None):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {'k': 3}
        super().__init__(RetrieverConfig())  # Initialize with default config
        
    def validate_config(self, config: RetrieverConfig) -> bool:
        """Validate retriever configuration."""
        try:
            required_fields = ['top_k', 'similarity_threshold']
            return all(hasattr(config, field) for field in required_fields)
        except Exception as e:
            logging.error(f"{COLORS['RED']}Config validation error: {str(e)}{COLORS['ENDC']}")
            return False

    def retrieve(self, query: str, documents: List[Document]) -> List[Document]:
        """Retrieve relevant documents using vector similarity."""
        try:
            results = self.vectorstore.similarity_search(
                query,
                **self.search_kwargs
            )
            logging.info(f"{COLORS['GREEN']}Retrieved {len(results)} documents{COLORS['ENDC']}")
            return results
        except Exception as e:
            logging.error(f"{COLORS['RED']}Retrieval error: {str(e)}{COLORS['ENDC']}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        try:
            self.vectorstore.add_documents(documents)
            logging.info(f"{COLORS['GREEN']}Added {len(documents)} documents to vector store{COLORS['ENDC']}")
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error adding documents: {str(e)}{COLORS['ENDC']}")
            raise
