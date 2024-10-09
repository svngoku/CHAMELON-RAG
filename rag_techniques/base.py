from pydantic import BaseModel, ConfigDict

class BaseComponent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def process(self, data):
        raise NotImplementedError("Subclasses should implement this!")

class BasePreprocessor(BaseComponent):
    def process(self, data):
        raise NotImplementedError("Subclasses should implement this!")


class BasePostprocessor(BaseComponent):
    def process(self, data):
        raise NotImplementedError("Subclasses should implement this!")


class BaseGenerator(BaseComponent):
    def generate(self, context, query):
        raise NotImplementedError("Subclasses should implement this!")


class BaseRetriever(BaseComponent):
    def retrieve(self, query):
        raise NotImplementedError("Subclasses should implement this!")
