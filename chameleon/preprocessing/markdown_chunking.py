from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from chameleon.base import BasePreprocessor
from typing import List, Union, Dict, Any
import logging
from langchain_core.documents import Document

class MarkdownChunking(BasePreprocessor):
    """Preprocessor for chunking markdown documents."""
    
    def __init__(self, name: str = "markdown_chunker", chunk_size: int = 500, chunk_overlap: int = 50):
        self.name = name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Define headers to split on with their respective levels
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def process(self, input_data: Union[str, List[Document], List[str]]) -> List[Document]:
        """Process the input data into chunks."""
        try:
            # If input is a list of Documents, extract text content
            if isinstance(input_data, list):
                if all(isinstance(doc, Document) for doc in input_data):
                    text = "\n\n".join(doc.page_content for doc in input_data)
                elif all(isinstance(s, str) for s in input_data):
                    text = "\n\n".join(input_data)
                else:
                    raise ValueError("Input list must contain either all Documents or all strings")
            else:
                text = input_data

            # Split text using markdown headers
            md_header_splits = self._markdown_splitter.split_text(text)
            
            # Further split into smaller chunks if needed
            chunks = []
            for doc in md_header_splits:
                sub_chunks = self._text_splitter.split_text(doc.page_content)
                for chunk in sub_chunks:
                    chunks.append(Document(
                        page_content=chunk,
                        metadata=doc.metadata
                    ))
            
            logging.info(f"Split input into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logging.error(f"Error during markdown chunking: {str(e)}")
            raise

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate preprocessor configuration."""
        required_fields = ['chunk_size', 'chunk_overlap']
        return all(field in config for field in required_fields)