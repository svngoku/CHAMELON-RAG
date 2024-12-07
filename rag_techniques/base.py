from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any
from langchain_core.documents import Document

class BaseComponent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def process(self, data):
        raise NotImplementedError("Subclasses should implement this!")

class BasePreprocessor(BaseComponent):
    def process(self, data):
        raise NotImplementedError("Subclasses should implement this!")


class BasePostprocessor(BaseComponent):
    def process(self, data):
        raise NotImplementedError("Subclasses should implement this!")


class BaseGenerator(BaseComponent):
    def process(self, context: str, query: str, chat_history: str = "") -> str:
        """Process the context and query to generate a response.
        
        Args:
            context: Retrieved context text
            query: User query
            chat_history: Optional chat history string
            
        Returns:
            Generated response text
        """
        raise NotImplementedError("Subclasses should implement this!")

    def generate(self, context: str, query: str):
        """Deprecated: Use process() instead."""
        raise NotImplementedError("Subclasses should implement this!")


class BaseRetriever(BaseComponent):
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents based on the query.
        
        Args:
            query: Search query string
            
        Returns:
            List of relevant documents
        """
        raise NotImplementedError("Subclasses should implement this!")


class BaseMemory(BaseComponent):
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load memory variables.
        
        Args:
            inputs: Input variables
            
        Returns:
            Dictionary containing memory variables (e.g., chat_history)
        """
        raise NotImplementedError("Subclasses should implement this!")
    
    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        """Save the context and output to memory.
        
        Args:
            inputs: Input context (e.g., user query)
            outputs: Output to save (e.g., generated response)
        """
        raise NotImplementedError("Subclasses should implement this!")
