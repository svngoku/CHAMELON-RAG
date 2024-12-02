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

    def run(self, query: str, data: Any) -> Dict[str, Any]:
        """Execute the full RAG pipeline"""
        logging.info("Starting RAGPipeline run method.")
        logging.info("Initial query: %s", query)

        # Convert documents to text if needed
        if hasattr(data, 'page_content'):
            data = data.page_content
        elif isinstance(data, list) and all(hasattr(d, 'page_content') for d in data):
            data = "\n\n".join(d.page_content for d in data)

        # Process through pipeline
        for preprocessor in self.preprocessors:
            data = preprocessor.process(data)
            
        context = self.retriever.get_relevant_documents(query)
        # Convert context to string if it's a list
        context_text = "\n".join([str(c) for c in context]) if isinstance(context, list) else context
        
        response = self.generator.generate(
            query=query, 
            context=context_text
        )

        return {
            "query": query,
            "context": context,
            "response": response
        }
