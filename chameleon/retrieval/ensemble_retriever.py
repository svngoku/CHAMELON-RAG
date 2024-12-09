from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from chameleon.base import BaseRetriever
from pydantic import BaseModel, Field, field_validator

class EnsembleRetriever(BaseModel, BaseRetriever):
    """Combines multiple retrievers with weighted scoring."""
    
    retrievers: List[BaseRetriever] = Field(
        ..., 
        description="List of retrievers to ensemble"
    )
    weights: List[float] = Field(
        ..., 
        description="Weights for each retriever"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    @field_validator('weights')
    def validate_weights(cls, v, values):
        """Validate weights sum to 1 and match number of retrievers."""
        if 'retrievers' in values and len(v) != len(values['retrievers']):
            raise ValueError("Number of weights must match number of retrievers")
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")
        return v

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Combine results from multiple retrievers using weighted scoring."""
        all_docs = []
        seen_contents = set()
        
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.get_relevant_documents(query)
            
            for doc in docs:
                if doc.page_content not in seen_contents:
                    doc.metadata['score'] = doc.metadata.get('score', 1.0) * weight
                    all_docs.append(doc)
                    seen_contents.add(doc.page_content)
        
        all_docs.sort(key=lambda x: x.metadata.get('score', 0.0), reverse=True)
        return all_docs