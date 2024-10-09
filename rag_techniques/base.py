from pydantic import BaseModel

class BaseComponent(BaseModel):
    def process(self, data):
        raise NotImplementedError("Subclasses should implement this!")

class BasePreprocessor(BaseComponent):
    pass

class BaseRetriever(BaseComponent):
    def retrieve(self, query):
        raise NotImplementedError("Subclasses should implement this!")

class BasePostprocessor(BaseComponent):
    pass

class BaseGenerator(BaseComponent):
    def generate(self, context, query):
        raise NotImplementedError("Subclasses should implement this!")
