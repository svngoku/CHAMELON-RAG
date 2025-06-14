from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from ..base import BasePreprocessor


class QueryTransformer(BasePreprocessor):
    """
    Transforms queries to improve retrieval performance based on LangChain best practices.
    
    Implements various query transformation techniques:
    - HyDE (Hypothetical Document Embeddings): Uses the LLM to generate a hypothetical document
      based on the query, then embeds that document instead of the original query
    - Query Expansion: Expands the query to include more contextual information
    - Multi-Query Generation: Generates multiple reformulations of the original query
    """
    
    def __init__(
        self, 
        llm: Optional[BaseLanguageModel] = None,
        technique: str = "hyde",
        num_queries: int = 3,
        **kwargs
    ):
        """
        Initialize the query transformer.
        
        Args:
            llm: Language model to use for transformations
            technique: Transformation technique to use (hyde, expansion, multi)
            num_queries: Number of queries to generate for multi-query technique
        """
        self.technique = technique
        self.num_queries = num_queries
        self.llm = llm or ChatOpenAI(temperature=0.0)
        
        # Set up transformation templates
        self.hyde_prompt = ChatPromptTemplate.from_template(
            "You are an expert search assistant. Based on the user question, write a "
            "detailed document that would contain the answer to the question.\n\n"
            "Question: {query}\n\n"
            "Document:"
        )
        
        self.expansion_prompt = ChatPromptTemplate.from_template(
            "You are an expert search assistant. Expand the following query to include "
            "more specific keywords and context that would help in retrieving relevant documents. "
            "Keep the expanded query concise (1-2 sentences).\n\n"
            "Original query: {query}\n\n"
            "Expanded query:"
        )
        
        self.multi_query_prompt = ChatPromptTemplate.from_template(
            "You are an expert search assistant. Generate {num_queries} different versions "
            "of the given query. Each version should focus on different aspects or use different "
            "terminology while preserving the original intent. Separate each query with a newline.\n\n"
            "Original query: {query}\n\n"
            "Generated queries:"
        )
    
    def process(self, query: str) -> Union[str, List[str]]:
        """
        Transform the input query using the specified technique.
        
        Args:
            query: The original query string
            
        Returns:
            Transformed query or list of queries
        """
        if self.technique == "hyde":
            return self._apply_hyde(query)
        elif self.technique == "expansion":
            return self._apply_expansion(query)
        elif self.technique == "multi":
            return self._apply_multi_query(query)
        else:
            raise ValueError(f"Unknown technique: {self.technique}")
    
    def _apply_hyde(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        chain = self.hyde_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def _apply_expansion(self, query: str) -> str:
        """Expand the query with additional context and keywords."""
        chain = self.expansion_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def _apply_multi_query(self, query: str) -> List[str]:
        """Generate multiple reformulations of the query."""
        chain = self.multi_query_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query, "num_queries": self.num_queries})
        
        # Split the result by newlines and filter out empty strings
        queries = [q.strip() for q in result.split("\n") if q.strip()]
        
        # Always include the original query
        if query not in queries:
            queries.append(query)
            
        return queries[:self.num_queries]  # Ensure we don't exceed num_queries