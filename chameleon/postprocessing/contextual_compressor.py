from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

from ..base import BasePostprocessor


class ContextualCompressor(BasePostprocessor):
    """
    Implements LangChain's Contextual Compression pattern to filter and compress retrieved documents.
    
    The contextual compressor:
    1. Takes retrieved documents and the original query
    2. Uses an LLM to filter out irrelevant parts of each document
    3. Returns more focused, relevant document chunks
    
    Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression
    """
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        compression_mode: str = "paragraph",
        min_relevance_score: float = 0.7
    ):
        """
        Initialize the contextual compressor.
        
        Args:
            llm: Language model to use for relevance assessment and compression
            compression_mode: How to compress documents ('paragraph', 'sentence', or 'full')
            min_relevance_score: Minimum relevance score for content to be included
        """
        self.llm = llm or ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
        self.compression_mode = compression_mode
        self.min_relevance_score = min_relevance_score
    
    def process(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Process retrieved documents to extract the most relevant parts.
        
        Args:
            query: The original query
            documents: The retrieved documents to compress
            
        Returns:
            List of compressed documents with only the most relevant parts
        """
        compressed_docs = []
        
        for doc in documents:
            # Split document by paragraphs if using paragraph mode
            if self.compression_mode == "paragraph":
                paragraphs = self._split_into_paragraphs(doc.page_content)
                relevant_parts = []
                
                for paragraph in paragraphs:
                    relevance = self._calculate_relevance(query, paragraph)
                    if relevance >= self.min_relevance_score:
                        relevant_parts.append(paragraph)
                
                if relevant_parts:
                    # Create a new document with only relevant paragraphs
                    compressed_content = "\n\n".join(relevant_parts)
                    compressed_docs.append(
                        Document(
                            page_content=compressed_content,
                            metadata={
                                **doc.metadata,
                                "compressed": True,
                                "original_length": len(doc.page_content),
                                "compressed_length": len(compressed_content)
                            }
                        )
                    )
                else:
                    # If no relevant paragraphs, still include the document but mark it
                    compressed_docs.append(
                        Document(
                            page_content=doc.page_content,
                            metadata={
                                **doc.metadata,
                                "compressed": False,
                                "relevance_warning": "No highly relevant content found"
                            }
                        )
                    )
            
            # For full document compression, use the LLM to extract relevant parts
            elif self.compression_mode == "full":
                compressed_content = self._extract_relevant_content(query, doc.page_content)
                compressed_docs.append(
                    Document(
                        page_content=compressed_content,
                        metadata={
                            **doc.metadata,
                            "compressed": True,
                            "original_length": len(doc.page_content),
                            "compressed_length": len(compressed_content)
                        }
                    )
                )
            
            # If not compressing, just pass through
            else:
                compressed_docs.append(doc)
        
        return compressed_docs
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = [p.strip() for p in text.split("\n\n")]
        return [p for p in paragraphs if p]  # Filter out empty paragraphs
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """
        Calculate relevance score of content to the query using the LLM.
        
        Uses a structured prompt to get the LLM to output a numeric score.
        """
        prompt = f"""On a scale of 0.0 to 1.0, rate how relevant the following text is to the query.
        Output only the numeric score without explanation.
        
        Query: {query}
        
        Text: {content}
        
        Relevance score (0.0 to 1.0):"""
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            # Extract the numeric score from the response
            score = float(response.split()[0]) if response else 0.0
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        except Exception:
            # If parsing fails, default to including the content (0.7 is just above threshold)
            return 0.7
    
    def _extract_relevant_content(self, query: str, content: str) -> str:
        """
        Use LLM to extract only the parts of the content relevant to the query.
        """
        prompt = f"""Extract ONLY the information from the following text that is directly relevant to answering this query.
        Maintain the exact wording from the original text for the extracted parts.
        Do not add any explanations or your own words.
        
        Query: {query}
        
        Text: {content}
        
        Relevant information:"""
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            return response
        except Exception:
            # If extraction fails, return the original content
            return content