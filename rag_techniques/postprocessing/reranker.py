from rag_techniques.base import BasePostprocessor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from typing import List
from langchain.docstore.document import Document
import logging


class ContextReranker(BasePostprocessor):
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.compressor = LLMChainExtractor.from_llm(self.llm)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=None
        )

    def process(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query"""
        try:
            # Score each document based on relevance
            scored_docs = []
            for doc in documents:
                relevance_score = self._calculate_relevance(query, doc.page_content)
                scored_docs.append((doc, relevance_score))
            
            # Sort by relevance score
            sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            
            # Return reranked documents
            return [doc for doc, _ in sorted_docs]
            
        except Exception as e:
            logging.error(f"Error in reranking: {str(e)}")
            return documents

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and document content"""
        prompt = f"""Rate the relevance of this document to the query on a scale of 0-1.
        Query: {query}
        Document: {content[:500]}  # Truncate for efficiency
        Score:"""
        
        try:
            response = self.llm.predict(prompt)
            return float(response.strip())
        except:
            return 0.0 