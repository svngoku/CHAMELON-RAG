from langchain_community.vectorstores import FAISS, Qdrant, PgVector, Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Union

class VectorDBFactory:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings()

    def create_vectorstore(self, data: Union[List[str], List[Document]], store_type: str = "faiss") -> Union[FAISS, Qdrant, PgVector, Pinecone]:
        documents = self._prepare_documents(data)
        if store_type == "faiss":
            return FAISS.from_documents(documents, self.embeddings)
        elif store_type == "qdrant":
            return Qdrant.from_documents(documents, self.embeddings)
        elif store_type == "pgvector":
            return PgVector.from_documents(documents, self.embeddings)
        elif store_type == "pinecone":
            return Pinecone.from_documents(documents, self.embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")

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
