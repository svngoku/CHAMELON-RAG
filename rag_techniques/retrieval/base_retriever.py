from pydantic import BaseModel, ConfigDict

class BaseComponent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class BaseRetriever(BaseComponent):
    def retrieve(self, query):
        raise NotImplementedError("Subclasses should implement this!")
