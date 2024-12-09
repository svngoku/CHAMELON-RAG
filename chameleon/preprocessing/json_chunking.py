from langchain_text_splitters import RecursiveJsonSplitter
from chameleon.base import BasePreprocessor
from chameleon.utils.logging_utils import COLORS
import logging
from typing import List, Union
import json

class JSONChunking(BasePreprocessor):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        convert_lists: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.convert_lists = convert_lists
        self.splitter = RecursiveJsonSplitter(
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def process(self, data: Union[str, dict]) -> List[str]:
        """Process JSON data into chunks."""
        logging.info("Starting JSON chunking process...")
        
        try:
            # Convert string to dict if needed
            if isinstance(data, str):
                data = json.loads(data)
                
            # Split the JSON data
            chunks = self.splitter.split_json(
                json_data=data,
                convert_lists=self.convert_lists
            )
            
            logging.info(f"Successfully split JSON into {len(chunks)} chunks")
            
            # Convert chunks back to strings
            text_chunks = [json.dumps(chunk) for chunk in chunks]
            
            return text_chunks
            
        except Exception as e:
            logging.error(f"Error in JSON chunking: {str(e)}")
            raise 