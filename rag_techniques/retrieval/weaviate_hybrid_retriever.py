from typing import List, Optional, Dict, Any
from langchain_community.retrievers import WeaviateHybridSearchRetriever
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import weaviate
import os
import logging
from rag_techniques.utils.logging_utils import COLORS

class WeaviateConfig(BaseModel):
    """Configuration for Weaviate connection and search."""
    url: str = Field(default="http://localhost:8080")
    api_key: Optional[str] = None
    index_name: str = Field(default="RAGDocuments")
    text_key: str = Field(default="text")
    attributes: List[str] = Field(default_factory=list)
    create_schema: bool = Field(default=True)

class WeaviateHybridRetriever(BaseModel):
    """Wrapper for Weaviate Hybrid Search retriever."""
    
    config: WeaviateConfig
    retriever: Optional[WeaviateHybridSearchRetriever] = None
    
    def initialize(self) -> None:
        """Initialize the Weaviate client and retriever."""
        try:
            # Setup authentication
            auth_config = (
                weaviate.AuthApiKey(api_key=self.config.api_key)
                if self.config.api_key
                else None
            )
            
            # Initialize Weaviate client
            client = weaviate.Client(
                url=self.config.url,
                auth_client_secret=auth_config,
                additional_headers={
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                } if os.getenv("OPENAI_API_KEY") else {}
            )
            
            logging.info(f"{COLORS['BLUE']}Initializing Weaviate Hybrid Search retriever{COLORS['ENDC']}")
            
            # Create retriever
            self.retriever = WeaviateHybridSearchRetriever(
                client=client,
                index_name=self.config.index_name,
                text_key=self.config.text_key,
                attributes=self.config.attributes,
                create_schema_if_missing=self.config.create_schema
            )
            
            logging.info(f"{COLORS['GREEN']}Successfully initialized Weaviate retriever{COLORS['ENDC']}")
            
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error initializing Weaviate: {str(e)}{COLORS['ENDC']}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the Weaviate index."""
        if not self.retriever:
            self.initialize()
        
        try:
            logging.info(f"{COLORS['BLUE']}Adding {len(documents)} documents to Weaviate{COLORS['ENDC']}")
            doc_ids = self.retriever.add_documents(documents)
            logging.info(f"{COLORS['GREEN']}Successfully added documents{COLORS['ENDC']}")
            return doc_ids
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error adding documents: {str(e)}{COLORS['ENDC']}")
            raise

    def get_relevant_documents(
        self, 
        query: str, 
        where_filter: Optional[Dict[str, Any]] = None,
        score: bool = False,
        k: int = 4
    ) -> List[Document]:
        """Retrieve relevant documents using hybrid search."""
        if not self.retriever:
            self.initialize()
            
        try:
            logging.info(f"{COLORS['BLUE']}Performing hybrid search for: {query}{COLORS['ENDC']}")
            
            # Prepare search parameters
            search_params = {"k": k}
            if where_filter:
                search_params["where_filter"] = where_filter
            if score:
                search_params["score"] = True
                
            # Perform search
            docs = self.retriever.get_relevant_documents(query, **search_params)
            
            logging.info(f"{COLORS['GREEN']}Retrieved {len(docs)} documents{COLORS['ENDC']}")
            return docs
            
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error during retrieval: {str(e)}{COLORS['ENDC']}")
            raise 