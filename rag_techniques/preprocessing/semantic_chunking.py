import logging
from rag_techniques.base import BasePreprocessor
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from typing import List, Union

class SemanticChunking(BasePreprocessor):
    def process(self, data: Union[str, List[str]]) -> List[str]:
        logging.info("Starting SemanticChunking process method.")

        try:
            # Initialize the semantic chunker with OpenAI embeddings
            logging.info("Initializing semantic chunker...")
            embeddings = OpenAIEmbeddings()
            chunker = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90
            )

            # Process the data using the semantic chunker
            logging.info("Processing data with semantic chunker...")
            if isinstance(data, str):
                chunks = chunker.create_documents([data])
            else:
                chunks = chunker.create_documents(data)

            # Extract text content from chunks
            processed_chunks = [chunk.page_content for chunk in chunks]

            logging.info(f"Data processed into {len(processed_chunks)} chunks")
            return processed_chunks

        except Exception as e:
            logging.error(f"Error in semantic chunking: {str(e)}")
            raise
