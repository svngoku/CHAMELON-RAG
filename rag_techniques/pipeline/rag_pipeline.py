# rag_techniques/pipeline.py
class RAGPipeline:
    def __init__(self):
        self.preprocessors = []
        self.retriever = None
        self.postprocessors = []
        self.generator = None
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

    def set_generator(self, generator):
        self.generator = generator

    def run(self, query, data):
        # Preprocess data
        for preprocessor in self.preprocessors:
            data = preprocessor.process(data)

        # Retrieve relevant information
        retrieved_data = self.retriever.retrieve(query)

        # Postprocess retrieved data
        for postprocessor in self.postprocessors:
            retrieved_data = postprocessor.process(retrieved_data)

        # Generate response
        response = self.generator.generate(retrieved_data, query)
        print(response)

        return response
