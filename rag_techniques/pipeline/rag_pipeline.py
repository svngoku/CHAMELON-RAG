import logging
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

    def run(self, query, data):
        logging.info("Starting RAGPipeline run method.")
        logging.info("Initial query: %s", query)

        # Apply each preprocessor to the data
        logging.info("Applying preprocessors...")
        for preprocessor in self.preprocessors:
            data = preprocessor.process(data)

        logging.info("Data after preprocessing: %s", data)

        # Retrieve relevant context using utility function
        logging.info("Retrieving context...")
        context = retrieve_context_per_question(query, self.retriever)

        logging.info("Context retrieved: %s", context)

        # Generate response using utility function
        logging.info("Generating response...")
        response_data = answer_question_from_context(query, context, self.generator)
        
        structured_output = {
            "query": query,
            "context": context,
            "response": response_data["answer"]
        }
        logging.info("Response generated: %s", response_data["answer"])
        logging.info("RAGPipeline run method completed.")

        return structured_output
