from pydantic import BaseModel

class BasePreprocessor:
    def process(self, data):
        raise NotImplementedError

class BaseRetriever:
    def retrieve(self, query):
        raise NotImplementedError

class BasePostprocessor:
    def process(self, retrieved_data):
        raise NotImplementedError

class BaseGenerator:
    def generate(self, context, query):
        raise NotImplementedError