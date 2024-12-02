import logging
from typing import Any, Dict
from rag_techniques.utils.utils import retrieve_context_per_question, answer_question_from_context

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGPipeline:
    def __init__(self):
        self.preprocessors = []
        self.retriever = None
        self.postprocessors = []
        self.generator = None
        self.reranker = None
        self.query_optimizer = None

    def add_preprocessor(self, preprocessor):
        self.preprocessors.append(preprocessor)
        return self

    def set_retriever(self, retriever):
        self.retriever = retriever
        return self

    def add_postprocessor(self, postprocessor):
        self.postprocessors.append(postprocessor)
        return self

    def set_generator(self, generator):
        self.generator = generator
        return self
        
    def set_reranker(self, reranker):
        self.reranker = reranker
        return self
        
    def set_query_optimizer(self, optimizer):
        self.query_optimizer = optimizer
        return self

    def run(self, query: str, data: Any) -> Dict[str, Any]:
        """Execute the full RAG pipeline with advanced techniques"""
        logging.info("Starting RAGPipeline run method.")
        
        # Optimize query if query optimizer is set
        if self.query_optimizer:
            query = self.query_optimizer.process(query)
            logging.info(f"Optimized query: {query}")

        # Convert input data to appropriate format
        processed_data = self._prepare_input_data(data)
            
        # Preprocess data
        for preprocessor in self.preprocessors:
            processed_data = preprocessor.process(processed_data)
            
        # Retrieve relevant documents
        context = self.retriever.get_relevant_documents(query)
        
        # Apply reranking if reranker is set
        if self.reranker:
            context = self.reranker.process(query, context)
            logging.info(f"Reranked context size: {len(context)}")
        
        # Apply postprocessing
        for postprocessor in self.postprocessors:
            context = postprocessor.process(context)
            
        # Convert context to string if it's a list
        context_text = "\n".join([str(c) for c in context]) if isinstance(context, list) else context
        
        # Generate response
        response = self.generator.generate(
            query=query, 
            context=context_text
        )

        return {
            "query": query,
            "context": context,
            "response": response,
            "metadata": {
                "preprocessors": len(self.preprocessors),
                "postprocessors": len(self.postprocessors),
                "reranking_applied": self.reranker is not None,
                "query_optimization_applied": self.query_optimizer is not None
            }
        }

    def _prepare_input_data(self, data: Any) -> str:
        """Convert input data to string format for preprocessing."""
        if isinstance(data, str):
            return data
        elif isinstance(data, list):
            if all(isinstance(d, str) for d in data):
                return "\n\n".join(data)
            elif all(hasattr(d, 'page_content') for d in data):
                return "\n\n".join(d.page_content for d in data)
        elif hasattr(data, 'page_content'):
            return data.page_content
        
        raise ValueError("Input data must be a string, list of strings, or Document object(s)")
