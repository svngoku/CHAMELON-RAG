from langchain.schema.runnable import Runnable
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from typing import List, Union, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
from langchain_core.documents import Document

class SemanticChunking(BaseModel, Runnable):
    """Runnable for semantically chunking text using embeddings."""
    
    breakpoint_threshold_type: str = Field(
        default="percentile",
        description="Type of threshold for breaking text"
    )
    breakpoint_threshold_amount: int = Field(
        default=90,
        ge=0,
        le=100,
        description="Threshold amount for breaking text"
    )
    embeddings_model: Optional[Any] = Field(
        default_factory=OpenAIEmbeddings,
        description="Embeddings model to use"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chunker = self._initialize_chunker()
        
    def _initialize_chunker(self) -> SemanticChunker:
        """Initialize the semantic chunker with current settings."""
        try:
            embeddings = (
                self.embeddings_model 
                if isinstance(self.embeddings_model, OpenAIEmbeddings) 
                else self.embeddings_model()
            )
            return SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.breakpoint_threshold_amount
            )
        except Exception as e:
            logging.error(f"Failed to initialize chunker: {str(e)}")
            raise
            
    def invoke(self, input_data: Union[str, List[str]]) -> List[Document]:
        """Process input data using semantic chunking.
        
        Args:
            input_data: String or list of strings to process
            
        Returns:
            List of Document objects
        """
        try:
            if isinstance(input_data, (str, list)):
                texts = [input_data] if isinstance(input_data, str) else input_data
                chunks = self.chunker.create_documents(texts)
                logging.info(f"Successfully created {len(chunks)} chunks")
                return chunks
            else:
                logging.warning(f"Unsupported input type: {type(input_data)}")
                return []
        except Exception as e:
            logging.error(f"Error during semantic chunking: {str(e)}")
            raise

    def batch(
        self, 
        inputs: List[Union[str, List[str]]], 
        config: Optional[Dict] = None
    ) -> List[List[Document]]:
        """Process multiple inputs in parallel."""
        return [self.invoke(input_data) for input_data in inputs]