from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import fitz  # PyMuPDF
from pathlib import Path
from chameleon.utils.logging_utils import COLORS
import textwrap

class DocumentLoader(BaseModel):
    """A flexible document loader that supports multiple file types."""
    
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    file_path: Optional[str] = Field(default=None)
    content: Optional[str] = Field(default=None)
    supported_formats: List[str] = Field(
        default=["pdf", "txt", "md"]
    )
    
    class Config:
        arbitrary_types_allowed = True

    def text_wrap(self, text: str, width: int = 120) -> str:
        """Wraps the input text to the specified width."""
        return textwrap.fill(text, width=width)

    def replace_t_with_space(self, documents: List[Document]) -> List[Document]:
        """Replaces all tab characters with spaces in document content."""
        for doc in documents:
            doc.page_content = doc.page_content.replace('\t', ' ')
        return documents

    def read_pdf_to_string(self, path: str) -> str:
        """Read a PDF document and return its content as a string."""
        doc = fitz.open(path)
        content = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            content += page.get_text()
        return content

    def load(self, source: Union[str, List[str]]) -> List[Document]:
        """Load documents from file path(s) or raw content."""
        try:
            if isinstance(source, str):
                if Path(source).exists():
                    self.file_path = source
                    return self._load_file()
                else:
                    self.content = source
                    return self._load_from_string()
            elif isinstance(source, list):
                return self._load_multiple_files(source)
            else:
                raise ValueError("Source must be a file path, list of file paths, or raw content string")
                
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error loading document: {str(e)}{COLORS['ENDC']}")
            raise

    def _load_file(self) -> List[Document]:
        """Load a single file based on its extension."""
        if not self.file_path:
            raise ValueError("No file path provided")
            
        file_extension = Path(self.file_path).suffix.lower()[1:]
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        logging.info(f"{COLORS['BLUE']}Loading {file_extension} file: {self.file_path}{COLORS['ENDC']}")
        
        if file_extension == "pdf":
            return self._load_pdf()
        elif file_extension == "txt":
            return self._load_text()
        elif file_extension == "md":
            return self._load_markdown()
            
    def _load_pdf(self) -> List[Document]:
        """Load a PDF file."""
        try:
            # Use the utility function to read PDF
            content = self.read_pdf_to_string(self.file_path)
            documents = [Document(page_content=content)]
            logging.info(f"{COLORS['GREEN']}Successfully loaded PDF{COLORS['ENDC']}")
            
            # Clean and split the documents
            documents = self.replace_t_with_space(documents)
            return self._split_documents(documents)
            
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error loading PDF: {str(e)}{COLORS['ENDC']}")
            raise

    def _load_text(self) -> List[Document]:
        """Load a text file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            documents = [Document(page_content=content)]
            documents = self.replace_t_with_space(documents)
            logging.info(f"{COLORS['GREEN']}Successfully loaded text file{COLORS['ENDC']}")
            return self._split_documents(documents)
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error loading text file: {str(e)}{COLORS['ENDC']}")
            raise

    def _load_markdown(self) -> List[Document]:
        """Load a markdown file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            documents = [Document(page_content=content)]
            documents = self.replace_t_with_space(documents)
            logging.info(f"{COLORS['GREEN']}Successfully loaded markdown file{COLORS['ENDC']}")
            return self._split_documents(documents)
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error loading markdown file: {str(e)}{COLORS['ENDC']}")
            raise

    def _load_from_string(self) -> List[Document]:
        """Load content from a string."""
        try:
            documents = [Document(page_content=self.content)]
            documents = self.replace_t_with_space(documents)
            logging.info(f"{COLORS['GREEN']}Successfully loaded content from string{COLORS['ENDC']}")
            return self._split_documents(documents)
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error loading string content: {str(e)}{COLORS['ENDC']}")
            raise

    def _load_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """Load multiple files and combine their documents."""
        all_documents = []
        for file_path in file_paths:
            self.file_path = file_path
            documents = self._load_file()
            all_documents.extend(documents)
        return all_documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
            split_docs = splitter.split_documents(documents)
            logging.info(f"{COLORS['GREEN']}Split into {len(split_docs)} chunks{COLORS['ENDC']}")
            return split_docs
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error splitting documents: {str(e)}{COLORS['ENDC']}")
            raise 