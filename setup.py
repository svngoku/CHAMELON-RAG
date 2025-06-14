from chameleon.pipeline.rag_pipeline import RAGPipeline
from chameleon.retrieval.simple_retriever import SimpleRetriever
from chameleon.retrieval.base_retriever import BaseRetriever
from chameleon.vector_db_factory import VectorDBFactory
from chameleon.loaders import FileLoader, DocumentLoader
from chameleon.base import RetrieverConfig
from chameleon.generation.llm_generator import LLMGenerator, GeneratorConfig
from chameleon.preprocessing.markdown_chunking import MarkdownChunking
from chameleon.utils.logging_utils import setup_colored_logger, COLORS
from chameleon.memory.memory_factory import MemoryFactory
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
import os
import logging
from datasets import load_dataset
from langsmith import traceable
from langchain_openai import OpenAIEmbeddings
from chameleon.vector_db_factory import VectorDBConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

african_history_dataset = load_dataset("Svngoku/African-History-Extra-11-30-24", split="train")

class PipelineFactory:
    """Factory class for creating pipeline components."""
    
    RAG_TYPES = {
        "naive": "Basic RAG without advanced techniques",
        "modular": "Modular RAG with swappable components",
        "advanced": "Advanced RAG with re-ranking and filtering"
    }
    
    @staticmethod
    def create_markdown_preprocessor() -> MarkdownChunking:
        """Create and configure the markdown preprocessor."""
        return MarkdownChunking()
    
    @classmethod
    def create_retriever(cls, documents: List[Document], **config) -> BaseRetriever:
        """Create and configure the retriever."""
        # Create vector store config with defaults that can be overridden
        default_config = {
            "store_type": "faiss",
            "embedding_model": "text-embedding-3-small",
            "collection_name": "default_collection",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "persist_directory": "vector_stores",
            # Add retriever-specific defaults
            "top_k": 4,
            "similarity_threshold": 0.7,
            "retrieval_type": "similarity",
            "reranking_enabled": False,
            "filtering_enabled": False
        }
        
        # Update defaults with any provided config
        final_config = {**default_config, **config}
        
        # Create persist directory if it doesn't exist
        if not os.path.exists(final_config["persist_directory"]):
            os.makedirs(final_config["persist_directory"])
        
        # Initialize vector store factory
        vector_factory = VectorDBFactory(config=VectorDBConfig(**final_config))
        
        # Create vector store
        vector_db = vector_factory.create_vectorstore(documents)
        
        # Create retriever with config
        retriever = SimpleRetriever(config=RetrieverConfig(**final_config))
        
        # Set vector store after initialization
        retriever.vectorstore = vector_db
        
        logging.info(f"{COLORS['BLUE']}Created retriever with {vector_factory.description}{COLORS['ENDC']}")
        return retriever


    @staticmethod
    def create_generator(
        model_name: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        temperature: float = 0.3,
        max_tokens: int = 8192,
        provider: str = "together",
        **kwargs
    ) -> LLMGenerator:
        """Create and configure the LLM generator with customizable parameters."""
        config = GeneratorConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            **kwargs
        )
        return LLMGenerator(config=config)
    
    @classmethod
    @traceable
    def create_pipeline(
        cls,
        documents: List[Document],
        title: Optional[str] = None,
        rag_type: str = "modular",
        memory_type: Optional[str] = "buffer",
        memory_config: Optional[Dict[str, Any]] = None,
        retriever_config: Optional[Dict[str, Any]] = None,
        generator_config: Optional[Dict[str, Any]] = None
    ) -> RAGPipeline:
        """Create and configure the RAG pipeline with specified technique and components."""
        if rag_type not in cls.RAG_TYPES:
            raise ValueError(f"Invalid RAG type. Choose from: {list(cls.RAG_TYPES.keys())}")
        
        # Create memory if specified
        memory = None
        if memory_type:
            memory = MemoryFactory(
                memory_type=memory_type,
                memory_config=memory_config or {}
            ).create_memory()
            logging.info(f"{COLORS['BLUE']}Created {memory_type} memory for pipeline{COLORS['ENDC']}")
        
        # Create pipeline title if not provided
        if title is None:
            title = f"CHAMELEON {rag_type.capitalize()} RAG Pipeline"
        
        return RAGPipeline(
            title=title,
            documents=documents,
            preprocessors=[cls.create_markdown_preprocessor()],
            retriever=cls.create_retriever(documents, **(retriever_config or {})),
            generator=cls.create_generator(**(generator_config or {})),
            memory=memory
        )
    
    @classmethod
    def load_test_data(cls) -> List[Document]:
        """Load test data from files."""
        loader = FileLoader()
        test_files = [
            os.path.join("data", f)
            for f in os.listdir("data")
            if f.endswith(".txt")
        ]
        documents = []
        for file_path in test_files:
            documents.extend(loader.load_text_file(file_path))
        return documents
    
    @classmethod
    def load_test_data_from_dataset(cls, max_documents: int = 50) -> List[Document]:
        """
        Load test data from the African History dataset with size limit.
        
        Args:
            max_documents: Maximum number of documents to load (default: 50)
        """
        documents = []
        count = 0
        
        for item in african_history_dataset:
            if count >= max_documents:
                break
                
            content = item["content"]
            
            # Split long content into smaller chunks to avoid token limits
            if len(content) > 2000:  # Roughly 500 tokens
                # Split into chunks of ~2000 characters
                chunks = [content[i:i+2000] for i in range(0, len(content), 1500)]  # 500 char overlap
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": f"african_history_{count}_chunk_{i}"}
                    ))
            else:
                documents.append(Document(
                    page_content=content,
                    metadata={"source": f"african_history_{count}"}
                ))
            
            count += 1
        
        logging.info(f"{COLORS['BLUE']}Loaded {len(documents)} document chunks from {count} source documents{COLORS['ENDC']}")
        return documents

if __name__ == "__main__":
    # Load documents (limit to 20 documents for testing to avoid token limits)
    documents = PipelineFactory.load_test_data_from_dataset(max_documents=10)

    # Create pipeline with documents
    pipeline = PipelineFactory.create_pipeline(
        documents=documents,
        title="My RAG Pipeline",
        rag_type="advanced",
        memory_type="buffer",
        retriever_config={
            "store_type": "faiss",
            "embedding_model": "text-embedding-3-small"
        },
        generator_config={
            "model_name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "temperature": 0.7,
            "max_tokens": 4096,
            "provider": "together"
        }
    )

    # Run a query
    result = pipeline.run("What is the significance of the Dahlak archipelago?")
    print(result['response'])

    