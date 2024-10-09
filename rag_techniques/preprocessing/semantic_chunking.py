from rag_techniques.base import BasePreprocessor
import time
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
from langchain_openai import OpenAIEmbeddings
from rag_techniques.utils.utils import read_pdf_to_string, retrieve_context_per_question, show_context
from langchain.vectorstores import FAISS

class SemanticChunkingRAG:
    """
    A class to handle the Semantic Chunking RAG process for document chunking and query retrieval.
    """

    def __init__(self, path, n_retrieved=2, embeddings=None, breakpoint_type: BreakpointThresholdType = "percentile",
                 breakpoint_amount=90):
        """
        Initializes the SemanticChunkingRAG by encoding the content using a semantic chunker.

        Args:
            path (str): Path to the PDF file to encode.
            n_retrieved (int): Number of chunks to retrieve for each query (default: 2).
            embeddings: Embedding model to use.
            breakpoint_type (str): Type of semantic breakpoint threshold.
            breakpoint_amount (float): Amount for the semantic breakpoint threshold.
        """
        print("\n--- Initializing Semantic Chunking RAG ---")
        # Read PDF to string
        content = read_pdf_to_string(path)

        # Use provided embeddings or initialize OpenAI embeddings
        self.embeddings = embeddings if embeddings else OpenAIEmbeddings()

        # Initialize the semantic chunker
        self.semantic_chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount
        )

        # Measure time for semantic chunking
        start_time = time.time()
        self.semantic_docs = self.semantic_chunker.create_documents([content])
        self.time_records = {'Chunking': time.time() - start_time}
        print(f"Semantic Chunking Time: {self.time_records['Chunking']:.2f} seconds")

        # Create a vector store and retriever from the semantic chunks
        self.semantic_vectorstore = FAISS.from_documents(self.semantic_docs, self.embeddings)
        self.semantic_retriever = self.semantic_vectorstore.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query):
        """
        Retrieves and displays the context for the given query.

        Args:
            query (str): The query to retrieve context for.

        Returns:
            tuple: The retrieval time.
        """
        # Measure time for semantic retrieval
        start_time = time.time()
        semantic_context = retrieve_context_per_question(query, self.semantic_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Semantic Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        # Display the retrieved context
        show_context(semantic_context)
        return self.time_records


class SemanticChunking(BasePreprocessor):
    def process(self, data):
        # Initialize the semantic chunker with OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        chunker = SemanticChunker(embeddings=embeddings, threshold_type=BreakpointThresholdType.SEMANTIC)

        # Process the data using the semantic chunker
        chunks = chunker.chunk(data)

        # Return the processed chunks
        return chunks
