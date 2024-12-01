import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from rag_techniques.utils.utils import retrieve_context_per_question, answer_question_from_context

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class PipelineComponents:
    preprocessors: List[Any]
    retriever: Any
    postprocessors: List[Any] 
    generator: Any

class RAGPipeline:
    def __init__(self):
        self._components = PipelineComponents(
            preprocessors=[],
            retriever=None,
            postprocessors=[],
            generator=None
        )

    def add_preprocessor(self, preprocessor):
        self._components.preprocessors.append(preprocessor)
        return self

    def set_retriever(self, retriever):
        self._components.retriever = retriever
        return self

    def add_postprocessor(self, postprocessor):
        self._components.postprocessors.append(postprocessor)
        return self

    def set_generator(self, generator):
        self._components.generator = generator
        return self

    def _preprocess_data(self, data: Any) -> Any:
        """Handle preprocessing step"""
        logging.info("Applying preprocessors...")
        for preprocessor in self._components.preprocessors:
            data = preprocessor.process(data)
        logging.info("Data after preprocessing: %s", data)
        return data

    def _retrieve_context(self, query: str) -> str:
        """Handle context retrieval step"""
        logging.info("Retrieving context...")
        context = retrieve_context_per_question(query, self._components.retriever)
        logging.info("Context retrieved: %s", context)
        return context

    def _generate_response(self, query: str, context: str) -> Dict[str, Any]:
        """Handle response generation step"""
        logging.info("Generating response...")
        response_data = answer_question_from_context(
            query, 
            context, 
            self._components.generator
        )
        logging.info("Response generated: %s", response_data["answer"])
        return response_data

    def run(self, query: str, data: Any) -> Dict[str, Any]:
        """Execute the full RAG pipeline"""
        logging.info("Starting RAGPipeline run method.")
        logging.info("Initial query: %s", query)

        processed_data = self._preprocess_data(data)
        context = self._retrieve_context(query)
        response_data = self._generate_response(query, context)

        structured_output = {
            "query": query,
            "context": context,
            "response": response_data["answer"]
        }
        
        logging.info("RAGPipeline run method completed.")
        return structured_output
