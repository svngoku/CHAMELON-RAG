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
    def process(self, context, query):
        """Process the context and query to generate a response."""
        raise NotImplementedError("Subclasses should implement this!")

    def generate(self, context, query):
        """Deprecated: Use process() instead."""
        raise NotImplementedError("Subclasses should implement this!")


class BaseRetriever(BaseComponent):
    def retrieve(self, query):
        raise NotImplementedError("Subclasses should implement this!")
