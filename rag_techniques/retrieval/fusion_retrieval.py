from rag_techniques.retrieval.base_retriever import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pydantic import Field
from typing import List, Union
import numpy as np


class FusionRetrieval(BaseRetriever):
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    k: int = Field(default=5)
    alpha: float = Field(default=0.5)
    vectorstore: Union[None, FAISS] = Field(default=None)
    bm25: Union[None, object] = Field(default=None)
    embeddings: OpenAIEmbeddings = Field(default_factory=OpenAIEmbeddings)

    def process(self, data: Union[List[str], List[Document]]):
        documents = self._prepare_documents(data)
        cleaned_texts = self._replace_t_with_space(documents)
        self.vectorstore = FAISS.from_documents(cleaned_texts, self.embeddings)
        self.bm25 = self._create_bm25_index(cleaned_texts)

    def _prepare_documents(self, data: Union[List[str], List[Document]]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        if isinstance(data[0], str):
            texts = []
            for text in data:
                texts.extend(text_splitter.split_text(text))
            return [Document(page_content=t) for t in texts]
        elif isinstance(data[0], Document):
            return text_splitter.split_documents(data)
        else:
            raise ValueError("Input must be a list of strings or a list of Document objects")

    def _replace_t_with_space(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            doc.page_content = doc.page_content.replace('\t', ' ')
        return documents


    def retrieve(self, query: str) -> List[Document]:
        if not self.vectorstore or not self.bm25:
            raise ValueError("Retriever has not been initialized with data. Call process() first.")
        
        return self._fusion_retrieval(query)
    def _fusion_retrieval(self, query: str) -> List[Document]:
        # Retrieve all documents and calculate scores
        all_docs = self.vectorstore.similarity_search("", k=self.vectorstore.index.ntotal)
        bm25_scores = self.bm25.get_scores(query.split())
        vector_results = self.vectorstore.similarity_search_with_score(query, k=len(all_docs))

        vector_scores = np.array([score for _, score in vector_results])
        vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

        combined_scores = self.alpha * vector_scores + (1 - self.alpha) * bm25_scores
        sorted_indices = np.argsort(combined_scores)[::-1]

        return [all_docs[i] for i in sorted_indices[:self.k]]
