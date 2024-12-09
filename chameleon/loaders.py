from langchain.docstore.document import Document
from typing import List
import os
from bs4 import BeautifulSoup
import PyPDF2

class FileLoader:
    def load_text_file(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return [Document(page_content=content)]

    def load_html_file(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            content = soup.get_text()
        return [Document(page_content=content)]

    def load_pdf_file(self, file_path: str) -> List[Document]:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            content = ""
            for page in reader.pages:
                content += page.extract_text()
        return [Document(page_content=content)]

    def load_files(self, file_paths: List[str]) -> List[Document]:
        documents = []
        for file_path in file_paths:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.txt':
                documents.extend(self.load_text_file(file_path))
            elif ext == '.html':
                documents.extend(self.load_html_file(file_path))
            elif ext == '.pdf':
                documents.extend(self.load_pdf_file(file_path))
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        return documents
