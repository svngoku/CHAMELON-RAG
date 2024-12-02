from rag_techniques.loaders.base import BaseLoader
from langchain.docstore.document import Document
from typing import List
import os
from bs4 import BeautifulSoup
import PyPDF2
from pydantic import Field
from rag_techniques.utils.logging_utils import COLORS
import logging

class FileLoader(BaseLoader):
    """Loader for specific file types (txt, html, pdf)."""
    
    supported_formats: List[str] = Field(
        default=[".txt", ".html", ".pdf"]
    )

    def load_text_file(self, file_path: str) -> List[Document]:
        """Load a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            logging.info(f"{COLORS['GREEN']}Successfully loaded text file{COLORS['ENDC']}")
            return [Document(page_content=content)]
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error loading text file: {str(e)}{COLORS['ENDC']}")
            raise

    def load_html_file(self, file_path: str) -> List[Document]:
        """Load an HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                content = soup.get_text()
            logging.info(f"{COLORS['GREEN']}Successfully loaded HTML file{COLORS['ENDC']}")
            return [Document(page_content=content)]
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error loading HTML file: {str(e)}{COLORS['ENDC']}")
            raise

    def load_pdf_file(self, file_path: str) -> List[Document]:
        """Load a PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = ""
                for page in reader.pages:
                    content += page.extract_text()
            logging.info(f"{COLORS['GREEN']}Successfully loaded PDF file{COLORS['ENDC']}")
            return [Document(page_content=content)]
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error loading PDF file: {str(e)}{COLORS['ENDC']}")
            raise

    def load(self, file_paths: List[str]) -> List[Document]:
        """Load multiple files."""
        documents = []
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported file type: {ext}")
                
            logging.info(f"{COLORS['BLUE']}Loading file: {file_path}{COLORS['ENDC']}")
            
            if ext == '.txt':
                documents.extend(self.load_text_file(file_path))
            elif ext == '.html':
                documents.extend(self.load_html_file(file_path))
            elif ext == '.pdf':
                documents.extend(self.load_pdf_file(file_path))
                
        return documents 