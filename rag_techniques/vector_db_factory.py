from typing import List, Dict, Any, Optional, Literal, get_args
from langchain_community.vectorstores import FAISS, Qdrant, Pinecone, Chroma, Weaviate
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.indexes import IndexingResult, SQLRecordManager, index
from pydantic import BaseModel, Field, computed_field
from pathlib import Path
import logging
from rag_techniques.utils.logging_utils import COLORS

# Define supported vector stores
VECTOR_STORE_ENGINE = Literal["faiss", "qdrant", "pinecone", "chroma", "weaviate"]

class VectorDBConfig(BaseModel):
    """Configuration for vector database settings."""
    store_type: VECTOR_STORE_ENGINE = Field(default="faiss")
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    embedding_model: str = Field(default="text-embedding-3-small")
    collection_name: str = Field(default="default_collection")
    persist_directory: Optional[Path] = Field(default=None)
    index_document: bool = Field(default=False)
    collection_metadata: Optional[Dict[str, str]] = None

class VectorDBFactory(BaseModel):
    """Factory for creating and managing vector stores with document indexing."""
    
    config: VectorDBConfig
    _record_manager: Optional[SQLRecordManager] = None
    
    @computed_field
    def description(self) -> str:
        """Get a description of the vector store configuration."""
        desc = f"{self.config.store_type}/{self.config.collection_name}"
        if self.config.persist_directory:
            desc += f" => store: {self.config.persist_directory}"
        if self.config.index_document and self._record_manager:
            desc += f" indexer: {self._record_manager}"
        return desc

    def initialize_record_manager(self) -> None:
        """Initialize the SQL record manager for document indexing."""
        if self.config.index_document and self.config.persist_directory:
            db_url = f"sqlite:///{self.config.persist_directory}/record_manager_cache.sql"
            namespace = f"{self.config.store_type}/{self.config.collection_name}"
            self._record_manager = SQLRecordManager(namespace, db_url=db_url)
            self._record_manager.create_schema()
            logging.info(f"{COLORS['BLUE']}Initialized record manager: {db_url}{COLORS['ENDC']}")

    def create_vectorstore(self, documents: List[Document]) -> Any:
        """Create and configure the vector store with documents."""
        try:
            embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
            
            if self.config.store_type == "chroma":
                store = Chroma(
                    embedding_function=embeddings,
                    persist_directory=str(self.config.persist_directory),
                    collection_name=self.config.collection_name,
                    collection_metadata=self.config.collection_metadata
                )
            else:
                store_classes = {
                    "faiss": FAISS,
                    "qdrant": Qdrant,
                    "pinecone": Pinecone,
                    "weaviate": Weaviate
                }
                store_class = store_classes.get(self.config.store_type)
                store = store_class.from_documents(documents, embeddings)

            if self.config.index_document:
                self.initialize_record_manager()
                
            logging.info(f"{COLORS['GREEN']}Created vector store: {self.description}{COLORS['ENDC']}")
            return store
            
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error creating vector store: {str(e)}{COLORS['ENDC']}")
            raise

    def add_documents(self, documents: List[Document]) -> IndexingResult | List[str]:
        """Add documents with optional deduplication."""
        try:
            if not self.config.index_document:
                return self.create_vectorstore(documents)
            else:
                assert self._record_manager
                store = self.create_vectorstore([])  # Create empty store
                
                info = index(
                    documents,
                    self._record_manager,
                    store,
                    cleanup="incremental",
                    source_id_key="source"
                )
                logging.info(f"{COLORS['GREEN']}Indexed {len(documents)} documents{COLORS['ENDC']}")
                return info
                
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error adding documents: {str(e)}{COLORS['ENDC']}")
            raise

