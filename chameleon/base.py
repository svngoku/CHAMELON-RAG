from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from langchain_core.documents import Document
from pydantic import BaseModel, Field

class RetrieverConfig(BaseModel):
    """Configuration for retriever components."""
    top_k: int = Field(default=3, gt=0, description="Number of documents to retrieve")
    similarity_threshold: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity score for retrieved documents"
    )
    retrieval_type: str = Field(
        default="similarity", 
        description="Type of retrieval method to use"
    )
    reranking_enabled: bool = Field(
        default=False,
        description="Whether to enable reranking of retrieved documents"
    )
    filtering_enabled: bool = Field(
        default=False,
        description="Whether to enable filtering of retrieved documents"
    )
    filtering_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold for filtering documents"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model to use for embeddings"
    )
    multi_query_enabled: bool = Field(
        default=False,
        description="Whether to use multi-query retrieval"
    )
    parent_document_enabled: bool = Field(
        default=False,
        description="Whether to use parent document retrieval"
    )
    store_type: Optional[str] = Field(
        default=None,
        description="Type of vector store to use (faiss, chroma, pinecone, etc.)"
    )

class GeneratorConfig(BaseModel):
    """Configuration for generator components."""
    model_config = {"protected_namespaces": ()}
    
    model_name: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, gt=0)
    streaming: bool = Field(default=False)
    provider: str = Field(default="openai")
    generation_type: str = Field(default="chat")
    system_prompt: Optional[str] = Field(default=None)
    response_format: Optional[Dict[str, Any]] = Field(default=None)

class MemoryConfig(BaseModel):
    """Configuration for memory components."""
    memory_type: str = Field(default="buffer")
    max_history: int = Field(default=10, gt=0)
    return_messages: bool = Field(default=True)
    redis_url: Optional[str] = None
    storage_type: str = Field(default="in_memory")

class PipelineConfig(BaseModel):
    """Configuration for the RAG pipeline."""
    rag_type: str = Field(default="modular")
    retriever_config: RetrieverConfig = Field(default_factory=RetrieverConfig)
    generator_config: GeneratorConfig = Field(default_factory=GeneratorConfig)
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    preprocessor_configs: List[Dict[str, Any]] = Field(default_factory=list)
    chain_type: str = Field(default="stuff")

@runtime_checkable
class Component(Protocol):
    """Protocol for all pipeline components."""
    def validate_config(self, config: BaseModel) -> bool:
        """Validate component configuration."""
        ...

class BaseRetriever(ABC):
    """Base class for retriever components."""
    
    def __init__(self, config: RetrieverConfig):
        self.config = config
        if not self.validate_config(config):
            raise ValueError("Invalid retriever configuration")
    
    @abstractmethod
    def validate_config(self, config: RetrieverConfig) -> bool:
        """Validate retriever configuration."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, documents: List[Document]) -> List[Document]:
        """Retrieve relevant documents."""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the retriever's index."""
        pass

class BaseGenerator(ABC):
    """Base class for generator components."""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
    
    @abstractmethod
    def generate(
        self, 
        query: str, 
        context: List[Document], 
        chat_history: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response from context.
        
        Args:
            query: The user query
            context: Retrieved documents to use as context
            chat_history: Optional conversation history
            additional_context: Optional additional context information
            
        Returns:
            Dictionary with generated response and metadata
        """
        pass
    
    def stream(
        self, 
        query: str, 
        context: List[Document],
        chat_history: Optional[List[Dict[str, Any]]] = None
    ):
        """Stream generation results (optional)."""
        raise NotImplementedError("Streaming not implemented for this generator")

class BaseMemory(ABC):
    """Base class for memory components."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
    
    @abstractmethod
    def add(self, query: str, response: Dict[str, Any]) -> None:
        """Add interaction to memory."""
        pass
    
    @abstractmethod
    def get(self, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve memory contents.
        
        Args:
            k: Optional limit on number of entries to return
            
        Returns:
            List of memory entries
        """
        pass
    
    def clear(self) -> None:
        """Clear memory (optional implementation)."""
        pass

class BasePreprocessor(ABC):
    """Base class for preprocessor components."""
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data."""
        pass

class BasePostprocessor(ABC):
    """Base class for postprocessor components."""
    
    @abstractmethod
    def process(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Process retrieved documents based on the query.
        
        Args:
            query: The original query
            documents: Retrieved documents to process
            
        Returns:
            Processed documents
        """
        pass