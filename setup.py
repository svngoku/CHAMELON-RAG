from chameleon.pipeline.rag_pipeline import RAGPipeline
from chameleon.retrieval.simple_retriever import SimpleRetriever
from chameleon.vector_db_factory import VectorDBFactory, VectorDBConfig
from chameleon.loaders.file_loader import FileLoader
from chameleon.loaders.document_loader import DocumentLoader
from chameleon.generation.llm_generator import LLMGenerator
from chameleon.preprocessing.markdown_chunking import MarkdownChunking
from chameleon.utils.logging_utils import setup_colored_logger, COLORS
from chameleon.memory.memory_factory import MemoryFactory
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
import os
import logging
from datasets import load_dataset
from langsmith import traceable
from chameleon.generation.advanced_generator import AdvancedGenerator
from pathlib import Path
from chameleon.base import GeneratorConfig, BaseGenerator
from chameleon.retrieval.simple_retriever import SimpleRetriever


african_history_dataset = load_dataset("Svngoku/African-History-Extra-11-30-24", split="train")

class PipelineFactory:
    """Factory class for creating pipeline components."""
    
    RAG_TYPES = {
        "naive": "Basic RAG without advanced techniques",
        "modular": "Modular RAG with swappable components",
        "advanced": "Advanced RAG with re-ranking and filtering"
    }
    
    _generator_types = {
        "simple": LLMGenerator,
        "advanced": AdvancedGenerator
    }
    
    @staticmethod
    def create_config(chunk_size: int = 500, 
                     chunk_overlap: int = 50, 
                     embedding_model: str = "text-embedding-3-small") -> VectorDBConfig:
        """Create vector database configuration with customizable parameters."""
        return VectorDBConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model
        )
    
    @staticmethod
    def create_markdown_preprocessor() -> MarkdownChunking:
        """Create and configure the Markdown preprocessor."""
        return MarkdownChunking(
            name="markdown_chunker",
            chunk_size=500,
            chunk_overlap=50
        )

    @classmethod
    def register_generator(cls, name: str, generator_class: Any) -> None:
        """Register a new generator type."""
        cls._generator_types[name] = generator_class
    
    @classmethod
    def create_generator(cls, generator_type: str = "simple", config: Optional[Dict[str, Any]] = None) -> BaseGenerator:
        """Create a generator of specified type with configuration."""
        if generator_type not in cls._generator_types:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        # Initialize empty config if None
        config = config or {}
        
        # Convert dictionary config to GeneratorConfig
        generator_config = GeneratorConfig(
            model_name=config.get('model_name', "Qwen/Qwen2.5-7B-Instruct-Turbo"),
            temperature=config.get('temperature', 0.8),
            max_tokens=config.get('max_tokens', 2500),
            streaming=config.get('streaming', False),
            provider=config.get('provider', "together")
        )
        
        # Create the generator with the config
        generator_class = cls._generator_types[generator_type]
        return generator_class(config=generator_config)

    @classmethod
    def create_retriever(cls, documents: List[Document], config: Optional[Dict[str, Any]] = None) -> SimpleRetriever:
        """Create and configure the retriever with vector store."""
        config = config or {}
        
        # Create default persist directory if it doesn't exist
        persist_dir = Path("data/vector_store")
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        vector_config = VectorDBConfig(
            store_type="faiss",
            chunk_size=config.get('chunk_size', 500),
            chunk_overlap=config.get('chunk_overlap', 50),
            embedding_model=config.get('embedding_model', "text-embedding-3-small"),
            collection_name=config.get('collection_name', "default_collection"),
            persist_directory=persist_dir  # Add persist directory
        )
        
        factory = VectorDBFactory(config=vector_config)
        vectorstore = factory.create_vectorstore(documents)
        
        return SimpleRetriever(
            vectorstore=vectorstore,
            search_kwargs={
                'k': config.get('top_k', 3),
                'score_threshold': config.get('similarity_threshold', 0.7)
            }
        )

    @classmethod
    def load_test_data(cls, file_paths: Optional[List[str]] = None) -> List[Document]:
        """Load and return the test dataset with validation."""
        if file_paths is None:
            file_paths = [
                "data/africanhistory1.txt",
                "data/africanhistory.txt"
            ]
        
        loader = DocumentLoader(
            chunk_size=1000,
            chunk_overlap=200,
            file_loader=FileLoader(supported_formats=[".txt", ".md"]),
        )
        
        # Validate file existence
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Test data file not found: {path}")
            
        return loader.load(file_paths)

    @classmethod
    def load_test_data_from_dataset(cls) -> List[Document]:
        """Load and return the test dataset from the African History dataset."""
        documents = []
        for item in african_history_dataset:
            documents.append(Document(page_content=item["content"]))
        return documents

    @classmethod
    @traceable
    def create_pipeline(cls, 
                       documents: List[Document], 
                       rag_type: str = "modular",
                       memory_type: Optional[str] = "buffer",
                       memory_config: Optional[Dict[str, Any]] = None,
                       retriever_config: Optional[Dict[str, Any]] = None,
                       generator_config: Optional[Dict[str, Any]] = None) -> RAGPipeline:
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

        # Configure components
        preprocessors = [cls.create_markdown_preprocessor()]
        retriever = cls.create_retriever(documents, config=retriever_config)
        generator = cls.create_generator(config=generator_config)

        return RAGPipeline(
            title=f"CHAMELEON {rag_type.capitalize()} RAG Pipeline",
            preprocessors=preprocessors,
            retriever=retriever,
            generator=generator,
            memory=memory
        )


    @classmethod
    def create_advanced_generator(cls, config: GeneratorConfig) -> AdvancedGenerator:
        """Create and configure an advanced generator."""
        return AdvancedGenerator(config=config)

if __name__ == '__main__':
    setup_colored_logger()
    documents = PipelineFactory.load_test_data_from_dataset()
    print(f"Loaded {len(documents)} documents from dataset")
    pipeline = PipelineFactory.create_pipeline(documents)
    
    # query = "Compare the emergence of Neolithic cultures in West Africa with those in the Nile Valley and Northern Horn of Africa?"
    # query = "Where is the Dahlak archipelago located? And what is it's significance?"
    query = "What was the role of women in the political landscape of the Kongo kingdom?"
    #query = "Who were the Khoe-San, and how were they categorized?"
    
    print(f"\n{COLORS['BLUE']}Running RAG Pipeline{COLORS['ENDC']}")
    print(f"{COLORS['YELLOW']}Query:{COLORS['ENDC']} {query}\n")
    
    result = pipeline.run(query, documents)
    
    print(f"\n{COLORS['GREEN']}Final Response:{COLORS['ENDC']}")
    print(result['response'])
    print("\n" + "="*80 + "\n")