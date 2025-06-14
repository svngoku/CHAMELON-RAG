from typing import List, Dict, Any, Optional, Generator, Union
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
import asyncio
import time

from ..base import (
    BaseRetriever, BaseGenerator, BaseMemory, BasePreprocessor, BasePostprocessor,
    PipelineConfig, RetrieverConfig, GeneratorConfig, MemoryConfig
)
from ..retrieval.advanced_retriever import AdvancedRetriever
from ..retrieval.multi_query_retriever import MultiQueryRetrieverWrapper
from ..retrieval.parent_document_retriever import ParentDocumentRetriever
from ..generation.advanced_generator import AdvancedGenerator
from ..preprocessing.query_transformer import QueryTransformer
from ..postprocessing.contextual_compressor import ContextualCompressor
from ..memory.entity_memory import EntityMemory
from ..tools.tool_executor import ToolExecutor
# from ..evaluation.rag_evaluator import RAGEvaluator  # Temporarily disabled due to RAGAS compatibility
from ..utils.logging_utils import setup_colored_logger


class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with latest LangChain features and best practices.
    
    Key improvements:
    - Query preprocessing with multiple transformation techniques
    - Multi-query retrieval for improved recall
    - Parent document retrieval for maintaining context
    - Contextual compression for better document focus
    - Entity-aware memory for tracking conversation entities
    - Tool integration for dynamic query handling
    - Built-in evaluation metrics
    """
    
    def __init__(
        self,
        title: str,
        documents: List[Document],
        config: Optional[PipelineConfig] = None,
        retriever: Optional[BaseRetriever] = None,
        generator: Optional[BaseGenerator] = None,
        memory: Optional[BaseMemory] = None,
        preprocessors: Optional[List[BasePreprocessor]] = None,
        postprocessors: Optional[List[BasePostprocessor]] = None,
        tools: Optional[List[Any]] = None,
        llm: Optional[BaseLanguageModel] = None,
        enable_evaluation: bool = False
    ):
        """
        Initialize the enhanced RAG pipeline.
        
        Args:
            title: Title of the pipeline
            documents: Initial documents for the pipeline
            config: Pipeline configuration
            retriever: Custom retriever to use
            generator: Custom generator to use
            memory: Custom memory to use
            preprocessors: List of preprocessors to apply
            postprocessors: List of postprocessors to apply
            tools: Optional tools to integrate
            llm: Language model to use
            enable_evaluation: Whether to enable built-in evaluation
        """
        self.title = title
        self.config = config or PipelineConfig()
        self.logger = setup_colored_logger()
        self.llm = llm or ChatOpenAI(temperature=0.0)
        
        # Initialize components
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        self.documents = self._preprocess_documents(documents)
        
        # Set up enhanced components
        self.retriever = retriever or self._create_retriever()
        self.generator = generator or self._create_generator()
        self.memory = memory or self._create_memory()
        
        # Add tools if provided
        self.tools = []
        if tools:
            self.tool_executor = ToolExecutor(tools=tools, llm=self.llm)
            self.preprocessors.append(self.tool_executor)
        
        # Set up evaluation if enabled
        self.evaluator = None  # RAGEvaluator(llm=self.llm) if enable_evaluation else None
        
        # Initialize retriever with documents
        if hasattr(self.retriever, 'add_documents'):
            self.retriever.add_documents(self.documents)
        
        # Validate chain type
        if self.config.chain_type not in ["stuff", "map_reduce", "refine", "map_rerank"]:
            raise ValueError(f"Unsupported chain type: {self.config.chain_type}")
        
        # Track pipeline usage
        self.query_count = 0
        self.response_times = []
    
    def _create_retriever(self) -> BaseRetriever:
        """Create an enhanced retriever based on configuration."""
        # Create base retriever
        base_retriever = AdvancedRetriever(self.config.retriever_config)
        
        # Wrap with multi-query retriever if specified
        if getattr(self.config.retriever_config, "multi_query_enabled", False):
            retriever = MultiQueryRetrieverWrapper(
                config=self.config.retriever_config,
                base_retriever=base_retriever,
                llm=self.llm
            )
        else:
            retriever = base_retriever
        
        # Wrap with parent document retriever if specified
        if getattr(self.config.retriever_config, "parent_document_enabled", False):
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            # Create a splitter for child documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            retriever = ParentDocumentRetriever(
                config=self.config.retriever_config,
                base_retriever=retriever,
                child_splitter=splitter
            )
        
        return retriever
    
    def _create_generator(self) -> BaseGenerator:
        """Create an enhanced generator based on configuration."""
        return AdvancedGenerator(self.config.generator_config)
    
    def _create_memory(self) -> Optional[BaseMemory]:
        """Create enhanced memory based on configuration."""
        if not self.config.memory_config:
            return None
            
        if self.config.memory_config.memory_type == "entity":
            return EntityMemory(
                config=self.config.memory_config,
                llm=self.llm
            )
        else:
            # Use standard memory adapter
            from chameleon.memory.memory_adapter import MemoryAdapter
            return MemoryAdapter(self.config.memory_config)
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the enhanced RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            Response dictionary with results and metadata
        """
        start_time = time.time()
        self.query_count += 1
        
        try:
            # Log pipeline execution
            self.logger.info(f"Running enhanced RAG pipeline: {self.title}")
            self.logger.info(f"Query: {query}")
            
            # Get conversation history and relevant entities
            history = None
            relevant_entities = []
            
            if self.memory:
                history = self._get_history()
                
                # Get relevant entities if using entity memory
                if isinstance(self.memory, EntityMemory):
                    relevant_entities = self.memory.get_relevant_entities(query)
            
            # Apply any query preprocessing (transformations, tool execution)
            query_info = {"query": query}
            for preprocessor in self.preprocessors:
                if isinstance(preprocessor, QueryTransformer):
                    transformed_query = preprocessor.process(query)
                    query_info["transformed_query"] = transformed_query
                    # Use the transformed query for retrieval
                    query = transformed_query if isinstance(transformed_query, str) else query
                elif isinstance(preprocessor, ToolExecutor):
                    tool_result = preprocessor.process(query)
                    if tool_result.get("tool_used"):
                        query_info["tool_result"] = tool_result
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query, self.documents)
            
            # Apply postprocessing to retrieved documents
            processed_docs = retrieved_docs
            for postprocessor in self.postprocessors:
                if isinstance(postprocessor, ContextualCompressor):
                    processed_docs = postprocessor.process(query, processed_docs)
            
            # Generate response based on chain type
            if self.config.chain_type == "stuff":
                result = self._run_stuff_chain(query, processed_docs, history, relevant_entities)
            elif self.config.chain_type == "map_reduce":
                result = self._run_map_reduce_chain(query, processed_docs, history, relevant_entities)
            elif self.config.chain_type == "refine":
                result = self._run_refine_chain(query, processed_docs, history, relevant_entities)
            else:  # map_rerank
                result = self._run_map_rerank_chain(query, processed_docs, history, relevant_entities)
            
            # Update memory if available
            if self.memory:
                self.memory.add(query, result)
                result['chat_history'] = self._get_history()
            
            # Add query info to result
            result.update(query_info)
            
            # Add execution metrics
            execution_time = time.time() - start_time
            self.response_times.append(execution_time)
            
            result['metadata']['execution_time'] = execution_time
            result['metadata']['query_count'] = self.query_count
            
            # Run evaluation if enabled
            if self.evaluator:
                evaluation = self.evaluator.evaluate_response(
                    query=query,
                    response=result['response'],
                    context_docs=processed_docs
                )
                result['evaluation'] = evaluation
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in enhanced RAG pipeline: {str(e)}")
            raise
    
    async def arun(self, query: str) -> Dict[str, Any]:
        """Run the RAG pipeline asynchronously."""
        try:
            # Apply preprocessing
            query_info = {"query": query}
            for preprocessor in self.preprocessors:
                if isinstance(preprocessor, QueryTransformer):
                    transformed_query = preprocessor.process(query)
                    query_info["transformed_query"] = transformed_query
                    query = transformed_query if isinstance(transformed_query, str) else query
                elif isinstance(preprocessor, ToolExecutor):
                    tool_result = preprocessor.process(query)
                    if tool_result.get("tool_used"):
                        query_info["tool_result"] = tool_result
            
            # Get history
            history = self._get_history()
            relevant_entities = []
            
            if isinstance(self.memory, EntityMemory):
                relevant_entities = self.memory.get_relevant_entities(query)
            
            # Retrieve documents
            retrieved_docs = self.retriever.retrieve(query, self.documents)
            
            # Apply postprocessing
            processed_docs = retrieved_docs
            for postprocessor in self.postprocessors:
                processed_docs = postprocessor.process(query, processed_docs)
            
            # Run chain asynchronously
            if self.config.chain_type == "stuff":
                result = await self._arun_stuff_chain(query, processed_docs, history, relevant_entities)
            elif self.config.chain_type == "map_reduce":
                result = await self._arun_map_reduce_chain(query, processed_docs, history, relevant_entities)
            elif self.config.chain_type == "refine":
                result = await self._arun_refine_chain(query, processed_docs, history, relevant_entities)
            else:
                result = await self._arun_map_rerank_chain(query, processed_docs, history, relevant_entities)
            
            # Update memory
            if self.memory:
                self.memory.add(query, result)
                result['chat_history'] = self._get_history()
            
            # Add query info
            result.update(query_info)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in async RAG pipeline: {str(e)}")
            raise
    
    async def stream(self, query: str) -> Generator[str, None, None]:
        """Stream responses from the RAG pipeline."""
        try:
            # Apply preprocessing
            for preprocessor in self.preprocessors:
                if isinstance(preprocessor, QueryTransformer):
                    transformed_query = preprocessor.process(query)
                    query = transformed_query if isinstance(transformed_query, str) else query
            
            # Retrieve documents
            retrieved_docs = self.retriever.retrieve(query, self.documents)
            
            # Apply postprocessing
            processed_docs = retrieved_docs
            for postprocessor in self.postprocessors:
                processed_docs = postprocessor.process(query, processed_docs)
            
            # Stream generation
            async for chunk in self.generator.stream(query, processed_docs):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Error in streaming RAG pipeline: {str(e)}")
            raise
    
    def _run_stuff_chain(
        self, 
        query: str, 
        documents: List[Document], 
        history: Optional[List[Dict[str, Any]]] = None,
        relevant_entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run stuff chain - process all documents together."""
        # Get entity information if relevant
        entity_context = ""
        if relevant_entities and isinstance(self.memory, EntityMemory):
            entity_info = [
                f"{entity}: {self.memory.get_entity_info(entity)}"
                for entity in relevant_entities
                if self.memory.get_entity_info(entity)
            ]
            if entity_info:
                entity_context = "Relevant entities from conversation history:\n" + "\n".join(entity_info)
        
        # Generate response with all context
        response = self.generator.generate(
            query=query,
            context=documents,
            chat_history=history,
            additional_context=entity_context if entity_context else None
        )
        
        return {
            'query': query,
            'response': response.get('response', ''),
            'context': self._format_context(documents),
            'documents': documents,
            'metadata': {
                'chain_type': 'stuff',
                'num_docs': len(documents),
                'entities_used': relevant_entities if relevant_entities else [],
                **response.get('metadata', {})
            }
        }
    
    def _run_map_reduce_chain(
        self, 
        query: str, 
        documents: List[Document], 
        history: Optional[List[Dict[str, Any]]] = None,
        relevant_entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run map-reduce chain - process documents in parallel and combine results."""
        # Map phase - process each document
        responses = []
        for doc in documents:
            response = self.generator.generate(query, [doc], chat_history=history)
            responses.append(response)
        
        # Reduce phase - combine responses
        combined_context = "\n\n".join(r.get('response', '') for r in responses)
        
        # Add entity information if relevant
        if relevant_entities and isinstance(self.memory, EntityMemory):
            entity_info = [
                f"{entity}: {self.memory.get_entity_info(entity)}"
                for entity in relevant_entities
                if self.memory.get_entity_info(entity)
            ]
            if entity_info:
                entity_context = "Relevant entities from conversation history:\n" + "\n".join(entity_info)
                combined_context = entity_context + "\n\n" + combined_context
        
        final_response = self.generator.generate(
            query, 
            [Document(page_content=combined_context)],
            chat_history=history
        )
        
        return {
            'query': query,
            'response': final_response.get('response', ''),
            'context': self._format_context(documents),
            'documents': documents,
            'metadata': {
                'chain_type': 'map_reduce',
                'num_docs': len(documents),
                'intermediate_steps': responses,
                'entities_used': relevant_entities if relevant_entities else [],
                **final_response.get('metadata', {})
            }
        }
    
    def _run_refine_chain(
        self, 
        query: str, 
        documents: List[Document], 
        history: Optional[List[Dict[str, Any]]] = None,
        relevant_entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run refine chain - iteratively refine response with each document."""
        if not documents:
            return {
                'query': query,
                'response': "No relevant documents found to answer the query.",
                'context': "",
                'documents': [],
                'metadata': {
                    'chain_type': 'refine',
                    'num_docs': 0
                }
            }
        
        # Add entity information if relevant
        entity_context = ""
        if relevant_entities and isinstance(self.memory, EntityMemory):
            entity_info = [
                f"{entity}: {self.memory.get_entity_info(entity)}"
                for entity in relevant_entities
                if self.memory.get_entity_info(entity)
            ]
            if entity_info:
                entity_context = "Relevant entities from conversation history:\n" + "\n".join(entity_info)
        
        # Initial response with first document and entity context
        initial_doc = documents[0]
        if entity_context:
            # Combine entity context with first document
            combined_content = f"{entity_context}\n\n{initial_doc.page_content}"
            initial_doc = Document(
                page_content=combined_content,
                metadata=initial_doc.metadata
            )
        
        current_response = self.generator.generate(query, [initial_doc], chat_history=history)
        
        # Refine with remaining documents
        for doc in documents[1:]:
            refine_query = f"""Given the following context and current answer, refine the answer:
            Context: {doc.page_content}
            Current Answer: {current_response.get('response', '')}
            Query: {query}"""
            
            current_response = self.generator.generate(refine_query, [doc], chat_history=None)
        
        return {
            'query': query,
            'response': current_response.get('response', ''),
            'context': self._format_context(documents),
            'documents': documents,
            'metadata': {
                'chain_type': 'refine',
                'num_docs': len(documents),
                'entities_used': relevant_entities if relevant_entities else [],
                **current_response.get('metadata', {})
            }
        }
    
    def _run_map_rerank_chain(
        self, 
        query: str, 
        documents: List[Document], 
        history: Optional[List[Dict[str, Any]]] = None,
        relevant_entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run map-rerank chain - generate responses for each document and select best."""
        # Get entity information
        entity_context = ""
        if relevant_entities and isinstance(self.memory, EntityMemory):
            entity_info = [
                f"{entity}: {self.memory.get_entity_info(entity)}"
                for entity in relevant_entities
                if self.memory.get_entity_info(entity)
            ]
            if entity_info:
                entity_context = "Relevant entities from conversation history:\n" + "\n".join(entity_info)
        
        # Generate response for each document
        responses = []
        for i, doc in enumerate(documents):
            # Add entity context to first document only
            if i == 0 and entity_context:
                combined_content = f"{entity_context}\n\n{doc.page_content}"
                doc = Document(
                    page_content=combined_content,
                    metadata=doc.metadata
                )
            
            response = self.generator.generate(query, [doc], chat_history=history)
            responses.append(response)
        
        # Rerank responses based on relevance and completeness
        rerank_query = f"""Query: {query}
        
        Rate each of the following responses on relevance, accuracy, and completeness.
        Choose the best response that most fully and accurately answers the query.
        
        Responses to rank:
        {chr(10).join([f"[{i+1}] {r.get('response', '')}" for i, r in enumerate(responses)])}
        
        Return the number of the best response:"""
        
        rerank_docs = [
            Document(page_content=r.get('response', ''), metadata={'index': i})
            for i, r in enumerate(responses)
        ]
        
        ranking_response = self.generator.generate(rerank_query, rerank_docs)
        
        # Parse the ranking response to get the best index
        try:
            ranking_text = ranking_response.get('response', '')
            # Extract the first number from the response
            import re
            match = re.search(r'\d+', ranking_text)
            if match:
                best_index = int(match.group()) - 1  # Convert to 0-based index
                if 0 <= best_index < len(responses):
                    best_response = responses[best_index]
                else:
                    best_response = responses[0]
            else:
                best_response = responses[0]
        except Exception:
            # Default to first response if parsing fails
            best_response = responses[0]
        
        return {
            'query': query,
            'response': best_response.get('response', ''),
            'context': self._format_context(documents),
            'documents': documents,
            'metadata': {
                'chain_type': 'map_rerank',
                'num_docs': len(documents),
                'candidate_responses': responses,
                'entities_used': relevant_entities if relevant_entities else [],
                **best_response.get('metadata', {})
            }
        }
    
    # Async versions of the chain runners
    async def _arun_stuff_chain(self, query, documents, history, relevant_entities):
        return self._run_stuff_chain(query, documents, history, relevant_entities)
    
    async def _arun_map_reduce_chain(self, query, documents, history, relevant_entities):
        return self._run_map_reduce_chain(query, documents, history, relevant_entities)
    
    async def _arun_refine_chain(self, query, documents, history, relevant_entities):
        return self._run_refine_chain(query, documents, history, relevant_entities)
    
    async def _arun_map_rerank_chain(self, query, documents, history, relevant_entities):
        return self._run_map_rerank_chain(query, documents, history, relevant_entities)
    
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline usage statistics."""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "title": self.title,
            "query_count": self.query_count,
            "document_count": len(self.documents),
            "average_response_time": avg_response_time,
            "chain_type": self.config.chain_type,
            "retriever_type": type(self.retriever).__name__,
            "generator_type": type(self.generator).__name__
        }