from rag_techniques.utils.utils import retrieve_context_per_question, answer_question_from_context

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
        # Apply each preprocessor to the data
        for preprocessor in self.preprocessors:
            data = preprocessor.process(data)

        # Retrieve relevant context using utility function
        context = retrieve_context_per_question(query, self.retriever)

        # Generate response using utility function
        response_data = answer_question_from_context(query, context, self.generator)
        
        structured_output = {
            "query": query,
            "context": context,
            "response": response_data["answer"]
        }
        return structured_output
