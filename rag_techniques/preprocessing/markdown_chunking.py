from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rag_techniques.base import BasePreprocessor
from typing import List, Optional, Tuple
import logging
from langchain_core.documents import Document
from pydantic import BaseModel, Field

class MarkdownChunking(BasePreprocessor, BaseModel):
    headers_to_split_on: List[Tuple[str, str]] = Field(
        default=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    return_each_line: bool = Field(default=False)
    strip_headers: bool = Field(default=True)
    markdown_splitter: Optional[MarkdownHeaderTextSplitter] = Field(default=None)
    text_splitter: Optional[RecursiveCharacterTextSplitter] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize splitters after parent initialization
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            return_each_line=self.return_each_line,
            strip_headers=self.strip_headers
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def process(self, markdown_text: str) -> List[Document]:
        """Process markdown text into chunks."""
        logging.info("Starting Markdown chunking process...")
        
        try:
            # First split by headers
            md_header_splits = self.markdown_splitter.split_text(markdown_text)
            logging.info(f"Split markdown into {len(md_header_splits)} header-based chunks")
            
            # Then apply character-level splitting
            final_splits = self.text_splitter.split_documents(md_header_splits)
            logging.info(f"Final number of chunks after size constraints: {len(final_splits)}")
            
            # Log some sample chunks for verification
            if final_splits:
                logging.info("Sample chunk with metadata:")
                sample = final_splits[0]
                logging.info(f"Content preview: {sample.page_content[:100]}...")
                logging.info(f"Metadata: {sample.metadata}")
            
            return final_splits
            
        except Exception as e:
            logging.error(f"Error in Markdown chunking: {str(e)}")
            raise 