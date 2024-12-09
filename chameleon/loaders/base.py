from pydantic import BaseModel
from typing import List
from langchain.docstore.document import Document

class BaseLoader(BaseModel):
    """Base class for all document loaders."""
    
    class Config:
        arbitrary_types_allowed = True

    def load(self, source) -> List[Document]:
        """Load documents from source."""
        raise NotImplementedError 