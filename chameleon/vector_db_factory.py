from typing import List, Dict, Any, Optional, Literal, get_args
from langchain_community.vectorstores import FAISS, Qdrant, Pinecone, Chroma, Weaviate
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.indexes import IndexingResult, SQLRecordManager, index
from pydantic import BaseModel, Field, computed_field
from pathlib import Path
import logging
from chameleon.utils.logging_utils import COLORS


VECTOR_STORE_CONFIGS = {
    "faiss": {
        "class": FAISS,
        "requires_persist": True,
    },
    "qdrant": {
        "class": Qdrant,
        "requires_persist": True,
    },
    "pinecone": {
        "class": Pinecone,
        "requires_persist": False,
    },
    "chroma": {
        "class": Chroma,
        "requires_persist": True,
    },
    "weaviate": {
        "class": Weaviate,
        "requires_persist": False,
    }
}

VECTOR_STORE_ENGINE = Literal["faiss", "qdrant", "pinecone", "chroma", "weaviate"]

class VectorDBConfig(BaseModel):
    """Configuration for vector database settings."""
    store_type: VECTOR_STORE_ENGINE = Field(
        default="faiss",
        description="Type of vector store to use"
    )
    chunk_size: int = Field(
        default=500, 
        gt=0,
        description="Size of text chunks for processing"
    )
    chunk_overlap: int = Field(
        default=50, 
        ge=0,
        description="Overlap between text chunks"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use"
    )
    collection_name: str = Field(
        default="default_collection",
        description="Name of the vector store collection"
    )
    persist_directory: Optional[Path] = Field(
        default=None,
        description="Directory to persist vector store"
    )
    index_document: bool = Field(
        default=False,
        description="Whether to enable document indexing"
    )
    collection_metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional metadata for the collection"
    )

    @computed_field
    def store_config(self) -> Dict[str, Any]:
        """Get configuration for the selected vector store."""
        return VECTOR_STORE_CONFIGS[self.store_type]

class VectorDBFactory(BaseModel):
    """Factory for creating and managing vector stores with document indexing."""
    
    config: VectorDBConfig
    _record_manager: Optional[SQLRecordManager] = None
    
    @property
    def description(self) -> str:
        """Get a description of the vector store configuration."""
        return (f"{self.config.store_type} vector store with "
                f"{self.config.embedding_model} embeddings "
                f"(collection: {self.config.collection_name})")

    def _validate_persistence_requirements(self) -> None:
        """Validate persistence requirements for the vector store."""
        if self.config.store_config["requires_persist"] and not self.config.persist_directory:
            raise ValueError(f"{self.config.store_type} requires a persist_directory")

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
            self._validate_persistence_requirements()
            embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
            
            store_class = self.config.store_config["class"]
            store_kwargs = {
                "embedding_function": embeddings,
                "collection_name": self.config.collection_name,
                "collection_metadata": self.config.collection_metadata
            }
            
            if self.config.persist_directory:
                store_kwargs["persist_directory"] = str(self.config.persist_directory)
            
            if self.config.store_type == "chroma":
                store = store_class(**store_kwargs)
                if documents:
                    # Add documents in batches for Chroma
                    self._add_documents_in_batches(store, documents)
            else:
                # For other vector stores, handle large document collections with batching
                if len(documents) > 100:  # Batch if more than 100 documents
                    logging.info(f"{COLORS['YELLOW']}Processing {len(documents)} documents in batches to avoid token limits{COLORS['ENDC']}")
                    
                    # Create initial store with first batch
                    batch_size = 50  # Conservative batch size
                    first_batch = documents[:batch_size]
                    store = store_class.from_documents(first_batch, embeddings)
                    
                    # Add remaining documents in batches
                    remaining_docs = documents[batch_size:]
                    self._add_documents_in_batches(store, remaining_docs, batch_size)
                else:
                    # Small document collection, create normally
                    store = store_class.from_documents(documents, embeddings)

            if self.config.index_document:
                self.initialize_record_manager()
                
            logging.info(f"{COLORS['GREEN']}Created {self.config.store_type} vector store "
                         f"with {self.config.embedding_model} embeddings{COLORS['ENDC']}")
            return store
            
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error creating vector store: {str(e)}{COLORS['ENDC']}")
            raise

    def _add_documents_in_batches(self, store: Any, documents: List[Document], batch_size: int = 50) -> None:
        """Add documents to vector store in batches to avoid token limits."""
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                logging.info(f"{COLORS['BLUE']}Processing batch {batch_num}/{total_batches} ({len(batch)} documents){COLORS['ENDC']}")
                store.add_documents(batch)
            except Exception as e:
                logging.error(f"{COLORS['RED']}Error processing batch {batch_num}: {str(e)}{COLORS['ENDC']}")
                # Try with smaller batch size if this batch fails
                if len(batch) > 10:
                    logging.info(f"{COLORS['YELLOW']}Retrying with smaller batch size{COLORS['ENDC']}")
                    smaller_batch_size = len(batch) // 2
                    self._add_documents_in_batches(store, batch, smaller_batch_size)
                else:
                    raise

    def add_documents(self, documents: List[Document]) -> IndexingResult | List[str]:
        """Add documents with optional deduplication."""
        try:
            if not self.config.index_document:
                return self.create_vectorstore(documents)
            else:
                assert self._record_manager, "Record manager not initialized"
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