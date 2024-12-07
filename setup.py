from rag_techniques.pipeline.rag_pipeline import RAGPipeline
from rag_techniques.retrieval.simple_retriever import SimpleRetriever
from rag_techniques.vector_db_factory import VectorDBFactory, VectorDBConfig
from rag_techniques.loaders import FileLoader, DocumentLoader
from rag_techniques.generation.llm_generator import LLMGenerator
from rag_techniques.preprocessing.markdown_chunking import MarkdownChunking
from rag_techniques.utils.logging_utils import setup_colored_logger, COLORS
from rag_techniques.memory.memory_factory import MemoryFactory
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
import os
import logging
from datasets import load_dataset
from langsmith import traceable

african_history_dataset = load_dataset("Svngoku/African-History-Extra-11-30-24", split="train")

class PipelineFactory:
    """Factory class for creating pipeline components."""
    
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

    @staticmethod
    def create_generator(
        provider: str = "together",
        model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> LLMGenerator:
        """Create and configure the LLM generator with customizable parameters."""
        return LLMGenerator(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

    @classmethod
    def create_retriever(cls, documents: List[Document]) -> SimpleRetriever:
        """Create and configure the retriever with vector store."""
        config = VectorDBConfig(
            store_type="faiss",
            chunk_size=500,
            chunk_overlap=50,
            embedding_model="text-embedding-3-small",
            collection_name="default_collection"
        )
        
        factory = VectorDBFactory(config=config)
        
        vectorstore = factory.create_vectorstore(documents)
        return SimpleRetriever(vectorstore=vectorstore)

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
    def create_pipeline(cls, 
                       documents: List[Document], 
                       memory_type: Optional[str] = "buffer",
                       memory_config: Optional[Dict[str, Any]] = None) -> RAGPipeline:
        """Create and configure the RAG pipeline with optional memory."""
        
        # Create memory if specified
        memory = None
        if memory_type:
            memory = MemoryFactory(
                memory_type=memory_type,
                memory_config=memory_config or {}
            ).create_memory()
            logging.info(f"{COLORS['BLUE']}Created {memory_type} memory for pipeline{COLORS['ENDC']}")

        return RAGPipeline(
            title="RAG Pipeline",
            preprocessors=[cls.create_markdown_preprocessor()],
            retriever=cls.create_retriever(documents),
            generator=cls.create_generator(),
            memory=memory
        )

if __name__ == '__main__':
    setup_colored_logger()
    documents = PipelineFactory.load_test_data_from_dataset()
    print(f"Loaded {len(documents)} documents from dataset")
    pipeline = PipelineFactory.create_pipeline(documents)
    
    # query = "Compare the emergence of Neolithic cultures in West Africa with those in the Nile Valley and Northern Horn of Africa?"
    # query = "Where is the Dahlak archipelago located? And what is it's significance?"
    #query = "What was the role of women in the political landscape of the Kongo kingdom?"
    query = "Who were the key figures that met in Nuremberg in 1652 and what was the significance of their meeting?"
    
    print(f"\n{COLORS['BLUE']}Running RAG Pipeline{COLORS['ENDC']}")
    print(f"{COLORS['YELLOW']}Query:{COLORS['ENDC']} {query}\n")
    
    result = pipeline.run(query, documents)
    
    print(f"\n{COLORS['GREEN']}Final Response:{COLORS['ENDC']}")
    print(result['response'])
    print("\n" + "="*80 + "\n")