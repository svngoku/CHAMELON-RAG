from rag_techniques.base import BasePreprocessor
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType
from langchain_openai import OpenAIEmbeddings


class SemanticChunking(BasePreprocessor):
    def process(self, data):
        # Initialize the semantic chunker with OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        chunker = SemanticChunker(embeddings=embeddings, threshold_type=BreakpointThresholdType.SEMANTIC)

        # Process the data using the semantic chunker
        chunks = chunker.chunk(data)

        # Return the processed chunks
        return chunks
