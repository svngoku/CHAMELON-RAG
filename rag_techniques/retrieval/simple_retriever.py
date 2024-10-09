from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from typing import List

class SimpleRetriever:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)
