from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangchainBaseRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

from ..base import BaseRetriever, RetrieverConfig
from ..preprocessing.query_transformer import QueryTransformer


class MultiQueryRetrieverWrapper(BaseRetriever):
    """
    Implements LangChain's MultiQueryRetriever pattern.
    
    This retriever:
    1. Generates multiple query formulations using an LLM
    2. Runs all queries against the underlying retriever
    3. Deduplicates and reranks the combined results
    
    Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/multi_query
    """
    
    def __init__(
        self, 
        config: RetrieverConfig,
        base_retriever: Optional[BaseRetriever] = None,
        llm: Optional[BaseLanguageModel] = None,
        num_queries: int = 3
    ):
        """
        Initialize the multi-query retriever.
        
        Args:
            config: Configuration for the retriever
            base_retriever: Underlying retriever to use for each query
            llm: Language model to use for query generation
            num_queries: Number of query variations to generate
        """
        super().__init__(config)
        self.base_retriever = base_retriever
        self.llm = llm or ChatOpenAI(temperature=0.0)
        self.num_queries = num_queries
        self.query_transformer = QueryTransformer(
            llm=self.llm,
            technique="multi",
            num_queries=self.num_queries
        )
        self.document_store = []
        
    def validate_config(self, config: RetrieverConfig) -> bool:
        """Validate retriever configuration."""
        if config.top_k < 1:
            return False
        return True
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the retriever."""
        self.document_store = documents
        if self.base_retriever and hasattr(self.base_retriever, 'add_documents'):
            self.base_retriever.add_documents(documents)
    
    def retrieve(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Retrieve documents using multiple query variations.
        
        Args:
            query: The original query
            documents: Documents to search in
            
        Returns:
            List of retrieved documents
        """
        # Update document store if needed
        if documents != self.document_store:
            self.add_documents(documents)
        
        if not self.document_store:
            return []
        
        # Generate multiple query variations
        query_variations = self.query_transformer.process(query)
        
        # Retrieve documents for each query
        all_docs = []
        for variation in query_variations:
            docs = self.base_retriever.retrieve(variation, self.document_store)
            all_docs.extend(docs)
        
        # Deduplicate documents (using page_content as key)
        unique_docs = {}
        for doc in all_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
        
        # Sort by relevance and take top_k
        # For now, we'll just return the first top_k unique documents
        return list(unique_docs.values())[:self.config.top_k]