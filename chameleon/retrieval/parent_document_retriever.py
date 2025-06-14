from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_core.documents import Document
from ..base import BaseRetriever, RetrieverConfig


class ParentDocumentRetriever(BaseRetriever):
    """
    A retriever that maintains parent-child relationships between documents.
    
    This retriever:
    1. Maintains links between child chunks and their parent documents
    2. Retrieves relevant chunks using the base retriever
    3. Returns the full parent documents for context preservation
    
    Based on LangChain's Parent Document Retriever pattern:
    https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever
    """
    
    def __init__(
        self,
        config: RetrieverConfig,
        base_retriever: BaseRetriever,
        child_splitter: Optional[Any] = None,
    ):
        """
        Initialize the parent document retriever.
        
        Args:
            config: Configuration for the retriever
            base_retriever: The underlying retriever to use for chunk retrieval
            child_splitter: Optional text splitter for creating child documents
        """
        super().__init__(config)
        self.base_retriever = base_retriever
        self.child_splitter = child_splitter
        
        # Store mappings between child and parent documents
        self.parent_documents = {}  # Original parent documents by ID
        self.child_to_parent_map = {}  # Mapping from child ID to parent ID
        self.child_documents = []  # All child documents for searching
    
    def validate_config(self, config: RetrieverConfig) -> bool:
        """Validate retriever configuration."""
        if config.top_k < 1:
            return False
        return True
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the retriever, splitting into child documents if needed.
        
        Args:
            documents: List of parent documents to add
        """
        # Store original parent documents
        for doc in documents:
            # Generate a unique ID for the parent if not present
            parent_id = str(hash(doc.page_content))
            self.parent_documents[parent_id] = doc
            
            # Create child documents if splitter is provided
            if self.child_splitter:
                child_docs = self.child_splitter.split_documents([doc])
                for child in child_docs:
                    # Add parent metadata and ID to child
                    child_id = str(hash(child.page_content))
                    child.metadata["parent_id"] = parent_id
                    
                    # Update mappings
                    self.child_to_parent_map[child_id] = parent_id
                    
                    # Add to child documents list
                    self.child_documents.append(child)
            else:
                # If no splitter, use document as its own child
                self.child_to_parent_map[parent_id] = parent_id
                self.child_documents.append(doc)
        
        # Add child documents to base retriever
        if hasattr(self.base_retriever, 'add_documents'):
            self.base_retriever.add_documents(self.child_documents)
    
    def retrieve(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Retrieve documents based on query, returning full parent documents.
        
        Args:
            query: The query string
            documents: Documents to search (usually ignored as we use stored documents)
            
        Returns:
            List of parent documents corresponding to the retrieved child documents
        """
        # Update documents if new ones provided
        if documents and documents != self.child_documents:
            self.add_documents(documents)
        
        # Retrieve child documents
        child_docs = self.base_retriever.retrieve(query, self.child_documents)
        
        # Collect unique parent IDs
        parent_ids = set()
        for doc in child_docs:
            # Get parent ID either from metadata or from mapping
            child_id = str(hash(doc.page_content))
            parent_id = doc.metadata.get("parent_id", self.child_to_parent_map.get(child_id))
            if parent_id:
                parent_ids.add(parent_id)
        
        # Get parent documents
        parent_docs = []
        for parent_id in parent_ids:
            if parent_id in self.parent_documents:
                parent_docs.append(self.parent_documents[parent_id])
        
        # Limit to top_k parents
        return parent_docs[:self.config.top_k]