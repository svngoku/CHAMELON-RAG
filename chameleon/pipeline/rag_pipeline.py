from typing import List, Dict, Any, Optional, Generator
from chameleon.base import (
    BaseRetriever, BaseGenerator, BaseMemory, BasePreprocessor,
    PipelineConfig, RetrieverConfig, GeneratorConfig, MemoryConfig
)
from chameleon.retrieval.advanced_retriever import AdvancedRetriever
from chameleon.generation.llm_generator import LLMGenerator
from langchain_core.documents import Document
from chameleon.utils.logging_utils import setup_colored_logger
import logging
import asyncio
from dataclasses import asdict

class RAGPipeline:
    """Flexible RAG pipeline supporting multiple techniques and chain types."""
    
    def __init__(
        self,
        title: str,
        documents: List[Document],
        config: Optional[PipelineConfig] = None,
        retriever: Optional[BaseRetriever] = None,
        generator: Optional[BaseGenerator] = None,
        memory: Optional[BaseMemory] = None,
        preprocessors: List[BasePreprocessor] = None
    ):
        self.title = title
        self.config = config or PipelineConfig()
        self.logger = setup_colored_logger()
        
        # Store and preprocess documents
        self.preprocessors = preprocessors or []
        self.documents = self._preprocess_documents(documents)
        
        # Initialize components based on RAG type
        self.retriever = retriever or self._create_retriever()
        self.generator = generator or self._create_generator()
        self.memory = memory
        
        # Initialize retriever with documents
        if hasattr(self.retriever, 'add_documents'):
            self.retriever.add_documents(self.documents)
        
        # Validate chain type
        if self.config.chain_type not in ["stuff", "map_reduce", "refine", "map_rerank"]:
            raise ValueError(f"Unsupported chain type: {self.config.chain_type}")
    
    def _create_retriever(self) -> BaseRetriever:
        """Create appropriate retriever based on RAG type."""
        if self.config.rag_type == "advanced":
            return AdvancedRetriever(self.config.retriever_config)
        # Add other retriever types as needed
        return AdvancedRetriever(self.config.retriever_config)
    
    def _create_generator(self) -> BaseGenerator:
        """Create appropriate generator based on RAG type."""
        if self.config.rag_type == "advanced":
            return LLMGenerator(self.config.generator_config)
        # Add other generator types as needed
        return LLMGenerator(self.config.generator_config)
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the RAG pipeline with the specified technique."""
        try:
            # Log pipeline execution
            self.logger.info(f"Running {self.config.rag_type} RAG pipeline: {self.title}")
            self.logger.info(f"Query: {query}")
            
            # Get conversation history if memory exists
            history = self._get_history()
            
            # Choose chain type and execute
            if self.config.chain_type == "stuff":
                result = self._run_stuff_chain(query, history)
            elif self.config.chain_type == "map_reduce":
                result = self._run_map_reduce_chain(query, history)
            elif self.config.chain_type == "refine":
                result = self._run_refine_chain(query, history)
            else:  # map_rerank
                result = self._run_map_rerank_chain(query, history)
            
            # Update memory if available
            if self.memory:
                self.memory.add(query, result)
                result['chat_history'] = self._get_history()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in RAG pipeline: {str(e)}")
            raise
    
    async def arun(self, query: str) -> Dict[str, Any]:
        """Run the RAG pipeline asynchronously."""
        try:
            history = self._get_history()
            
            # Run chain asynchronously
            if self.config.chain_type == "stuff":
                result = await self._arun_stuff_chain(query, history)
            elif self.config.chain_type == "map_reduce":
                result = await self._arun_map_reduce_chain(query, history)
            elif self.config.chain_type == "refine":
                result = await self._arun_refine_chain(query, history)
            else:
                result = await self._arun_map_rerank_chain(query, history)
            
            if self.memory:
                self.memory.add(query, result)
                result['chat_history'] = self._get_history()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in async RAG pipeline: {str(e)}")
            raise
    
    async def stream(self, query: str) -> Generator[str, None, None]:
        """Stream responses from the RAG pipeline."""
        try:
            retrieved_docs = self.retriever.retrieve(query, self.documents)
            
            async for chunk in self.generator.stream(query, retrieved_docs):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Error in streaming RAG pipeline: {str(e)}")
            raise
    
    def _run_stuff_chain(self, query: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run stuff chain - process all documents together."""
        retrieved_docs = self.retriever.retrieve(query, self.documents)
        response = self.generator.generate(query, retrieved_docs)
        
        return {
            'query': query,
            'response': response.get('response', ''),
            'context': self._format_context(retrieved_docs),
            'documents': retrieved_docs,
            'metadata': {
                'chain_type': 'stuff',
                'num_docs': len(retrieved_docs),
                **response.get('metadata', {})
            }
        }
    
    def _run_map_reduce_chain(self, query: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run map-reduce chain - process documents in parallel and combine results."""
        retrieved_docs = self.retriever.retrieve(query, self.documents)
        
        # Map phase - process each document
        responses = []
        for doc in retrieved_docs:
            response = self.generator.generate(query, [doc])
            responses.append(response)
        
        # Reduce phase - combine responses
        combined_context = "\n\n".join(r.get('response', '') for r in responses)
        final_response = self.generator.generate(query, [Document(page_content=combined_context)])
        
        return {
            'query': query,
            'response': final_response.get('response', ''),
            'context': self._format_context(retrieved_docs),
            'documents': retrieved_docs,
            'metadata': {
                'chain_type': 'map_reduce',
                'num_docs': len(retrieved_docs),
                'intermediate_steps': responses,
                **final_response.get('metadata', {})
            }
        }
    
    def _run_refine_chain(self, query: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run refine chain - iteratively refine response with each document."""
        retrieved_docs = self.retriever.retrieve(query, self.documents)
        
        # Initial response with first document
        current_response = self.generator.generate(query, [retrieved_docs[0]])
        
        # Refine with remaining documents
        for doc in retrieved_docs[1:]:
            refine_query = f"""Given the following context and current answer, refine the answer:
            Context: {doc.page_content}
            Current Answer: {current_response.get('response', '')}
            Query: {query}"""
            
            current_response = self.generator.generate(refine_query, [doc])
        
        return {
            'query': query,
            'response': current_response.get('response', ''),
            'context': self._format_context(retrieved_docs),
            'documents': retrieved_docs,
            'metadata': {
                'chain_type': 'refine',
                'num_docs': len(retrieved_docs),
                **current_response.get('metadata', {})
            }
        }
    
    def _run_map_rerank_chain(self, query: str, history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run map-rerank chain - generate responses for each document and select best."""
        retrieved_docs = self.retriever.retrieve(query, self.documents)
        
        # Generate response for each document
        responses = []
        for doc in retrieved_docs:
            response = self.generator.generate(query, [doc])
            responses.append(response)
        
        # Rerank responses based on relevance
        rerank_query = f"Query: {query}\nSelect the most relevant and complete response:"
        rerank_docs = [
            Document(page_content=r.get('response', ''), metadata={'original_response': r})
            for r in responses
        ]
        
        best_response = self.generator.generate(rerank_query, rerank_docs)
        
        return {
            'query': query,
            'response': best_response.get('response', ''),
            'context': self._format_context(retrieved_docs),
            'documents': retrieved_docs,
            'metadata': {
                'chain_type': 'map_rerank',
                'num_docs': len(retrieved_docs),
                'candidate_responses': responses,
                **best_response.get('metadata', {})
            }
        }
    
    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Apply preprocessing steps to documents."""
        processed = documents
        for preprocessor in self.preprocessors:
            processed = preprocessor.process(processed)
        return processed
    
    def _get_history(self) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history if memory is available."""
        try:
            return self.memory.get() if self.memory else None
        except Exception as e:
            self.logger.warning(f"Error getting memory: {str(e)}")
            return None
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into a context string."""
        return "\n\n".join(doc.page_content for doc in documents)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the pipeline."""
        processed_docs = self._preprocess_documents(documents)
        self.documents.extend(processed_docs)
        
        # Update retriever if it supports document addition
        if hasattr(self.retriever, 'add_documents'):
            self.retriever.add_documents(processed_docs)