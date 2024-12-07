from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rag_techniques.base import BasePreprocessor
from typing import List, Union, Any
import logging
from langchain_core.documents import Document
from pydantic import Field, PrivateAttr

class MarkdownChunking(BasePreprocessor):
    name: str = Field(default="markdown_chunker", description="Name of the preprocessor")
    chunk_size: int = Field(default=500, gt=0, description="Size of each chunk")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap between chunks")
    
    # Private attributes that won't be validated by Pydantic
    _markdown_splitter: Any = PrivateAttr(default=None)
    _text_splitter: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any = None) -> None:
        """Initialize after Pydantic validation."""
        super().model_post_init(context)
        
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