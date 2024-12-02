from pydantic import BaseModel, Field
from typing import List, Optional, Union
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from pathlib import Path
from rag_techniques.utils.logging_utils import COLORS
import textwrap
from bs4 import BeautifulSoup
import PyPDF2

class DocumentLoader(BaseModel):
    """Advanced document loader with chunking and preprocessing capabilities."""
    
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    supported_formats: List[str] = Field(
        default=[".txt", ".html", ".pdf", ".md"]
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

    def load(self, source: Union[str, List[str]]) -> List[Document]:
        """Load and process documents from various sources."""
        try:
            if isinstance(source, str):
                return self._load_file(source)
            elif isinstance(source, list):
                return self._load_multiple_files(source)
            else:
                raise ValueError("Source must be a file path or list of file paths")
                
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error loading document: {str(e)}{COLORS['ENDC']}")
            raise

    def _load_file(self, file_path: str) -> List[Document]:
        """Load a single file based on its extension."""
        ext = Path(file_path).suffix.lower()
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported file type: {ext}")
        
        logging.info(f"{COLORS['BLUE']}Loading file: {file_path}{COLORS['ENDC']}")
        
        if ext == '.txt':
            return self._load_text_file(file_path)
        elif ext == '.html':
            return self._load_html_file(file_path)
        elif ext == '.pdf':
            return self._load_pdf_file(file_path)
        elif ext == '.md':
            return self._load_markdown_file(file_path)

    def _load_text_file(self, file_path: str) -> List[Document]:
        """Load a text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        documents = [Document(page_content=content)]
        return self._split_documents(self.replace_t_with_space(documents))

    def _load_html_file(self, file_path: str) -> List[Document]:
        """Load an HTML file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            content = soup.get_text()
        documents = [Document(page_content=content)]
        return self._split_documents(self.replace_t_with_space(documents))

    def _load_pdf_file(self, file_path: str) -> List[Document]:
        """Load a PDF file."""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            content = ""
            for page in reader.pages:
                content += page.extract_text()
        documents = [Document(page_content=content)]
        return self._split_documents(self.replace_t_with_space(documents))

    def _load_markdown_file(self, file_path: str) -> List[Document]:
        """Load a markdown file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        documents = [Document(page_content=content)]
        return self._split_documents(self.replace_t_with_space(documents))

    def _load_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """Load multiple files and combine their documents."""
        all_documents = []
        for file_path in file_paths:
            documents = self._load_file(file_path)
            all_documents.extend(documents)
        return all_documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        split_docs = splitter.split_documents(documents)
        logging.info(f"{COLORS['GREEN']}Split into {len(split_docs)} chunks{COLORS['ENDC']}")
        return split_docs