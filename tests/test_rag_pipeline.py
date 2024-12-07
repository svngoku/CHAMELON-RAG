import unittest
from rag_techniques.pipeline.rag_pipeline import RAGPipeline
from rag_techniques.utils.logging_utils import setup_colored_logger, COLORS
from langchain_core.documents import Document
import os
import logging
from setup import PipelineFactory
from typing import List

class TestRAGPipeline(unittest.TestCase):
    """Test suite for RAG pipeline functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures with environment-aware logging."""
        env = os.getenv('TEST_ENV', 'development')
        log_level = logging.DEBUG if env == 'development' else logging.INFO
        cls.logger = setup_colored_logger(level=log_level)
        cls.test_documents = PipelineFactory.load_test_data()
        
    def setUp(self):
        """Set up test-level fixtures."""
        self.pipeline = PipelineFactory.create_pipeline(
            documents=self.test_documents,
            memory_type="buffer",
            memory_config={"return_messages": True}
        )

    def test_pipeline_run(self):
        """Test basic pipeline functionality."""
        query = "Compare the emergence of Neolithic cultures in West Africa with those in the Nile Valley and Northern Horn of Africa?"
        self.logger.info(f"{COLORS['BLUE']}Testing query: {query}{COLORS['ENDC']}")
        
        response = self.pipeline.run(query, self.test_documents)
        
        self._print_results(query, response)
        self._verify_response(response)

    def _print_results(self, query: str, response: dict):
        """Print test results in a formatted way."""
        print(f"\n{COLORS['GREEN']}=== Test Results ==={COLORS['ENDC']}")
        print(f"{COLORS['BLUE']}Query:{COLORS['ENDC']} {query}")
        print(f"{COLORS['GREEN']}Response:{COLORS['ENDC']} {response['response']}")
        print(f"{COLORS['YELLOW']}Context:{COLORS['ENDC']} {response['context'][:200]}...")

    def _verify_response(self, response: dict):
        """Verify the response meets all requirements."""
        self.assertIsInstance(response, dict)
        self.assertIn('query', response)
        self.assertIn('context', response)
        self.assertIn('response', response)
        self.assertIsInstance(response['response'], str)
        self.assertTrue(len(response['response']) > 0, "Response should not be empty")

    def test_pipeline_components(self):
        """Test if pipeline components are properly initialized."""
        self.assertIsNotNone(self.pipeline.preprocessors)
        self.assertIsNotNone(self.pipeline.retriever)
        self.assertIsNotNone(self.pipeline.generator)
        self.assertIsNotNone(self.pipeline.memory)
        self.assertEqual(len(self.pipeline.preprocessors), 1)

    def test_markdown_preprocessing(self):
        """Test Markdown preprocessing functionality."""
        markdown_text = """# Introduction
        
        ## Background
        This is some background information.
        More background details here.
        
        ## Methods
        Here are the methods we used.
        
        ### Data Collection
        Details about data collection.
        """
        
        preprocessor = PipelineFactory.create_markdown_preprocessor()
        chunks = preprocessor.process(markdown_text)
        
        self._verify_markdown_chunks(chunks)
        self._print_markdown_chunks(chunks)

    def _verify_markdown_chunks(self, chunks: List[Document]):
        """Verify markdown chunks meet requirements."""
        self.assertTrue(len(chunks) > 0, "Should produce at least one chunk")
        for chunk in chunks:
            self.assertIsInstance(chunk, Document)
            self.assertTrue(hasattr(chunk, 'page_content'))
            self.assertTrue(hasattr(chunk, 'metadata'))

    def _print_markdown_chunks(self, chunks: List[Document]):
        """Print markdown chunks for inspection."""
        self.logger.info(f"{COLORS['GREEN']}Markdown Chunks:{COLORS['ENDC']}")
        for i, chunk in enumerate(chunks):
            self.logger.info(f"{COLORS['BLUE']}Chunk {i}:{COLORS['ENDC']}")
            self.logger.info(f"Content: {chunk.page_content[:100]}...")
            self.logger.info(f"Headers: {chunk.metadata}")

    def test_pipeline_with_documents(self):
        """Test pipeline with Document objects as input."""
        query = "What are the key characteristics of early African civilizations?"
        self.logger.info(f"{COLORS['BLUE']}Testing query with documents: {query}{COLORS['ENDC']}")
        
        response = self.pipeline.run(query, self.test_documents)
        
        self._print_results(query, response)
        self._verify_response(response)

    """Test suite for RAG pipeline functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures with environment-aware logging."""
        env = os.getenv('TEST_ENV', 'development')
        log_level = logging.DEBUG if env == 'development' else logging.INFO
        cls.logger = setup_colored_logger(level=log_level)
        cls.test_documents = PipelineFactory.load_test_data()
        
    def setUp(self):
        """Set up test-level fixtures."""
        self.pipeline = PipelineFactory.create_pipeline(
            documents=self.test_documents,
            memory_type="buffer",  # Use buffer memory for testing
            memory_config={"return_messages": True}
        )

    def test_pipeline_run(self):
        """Test basic pipeline functionality."""
        query = "Compare the emergence of Neolithic cultures in West Africa with those in the Nile Valley and Northern Horn of Africa?"
        self.logger.info(f"{COLORS['BLUE']}Testing query: {query}{COLORS['ENDC']}")
        
        response = self.pipeline.run(query, self.test_documents)
        
        self._print_results(query, response)
        self._verify_response(response)

    def test_memory_retention(self):
        """Test if memory retains conversation history."""
        # First query
        query1 = "What are the key characteristics of early African civilizations?"
        response1 = self.pipeline.run(query1, self.test_documents)
        self._verify_response(response1)
        
        # Follow-up query
        query2 = "Can you elaborate on their agricultural practices?"
        response2 = self.pipeline.run(query2, self.test_documents)
        self._verify_response(response2)
        
        # Verify chat history exists
        self.assertIn("chat_history", response2)
        self.assertTrue(response2["chat_history"])

    def _verify_response(self, response: dict):
        """Verify the response meets all requirements."""
        self.assertIsInstance(response, dict)
        self.assertIn('query', response)
        self.assertIn('context', response)
        self.assertIn('response', response)
        self.assertIsInstance(response['response'], str)
        self.assertTrue(len(response['response']) > 0)

    def _print_results(self, query: str, response: dict):
        """Print test results in a formatted way."""
        print(f"\n{COLORS['GREEN']}=== Test Results ==={COLORS['ENDC']}")
        print(f"{COLORS['BLUE']}Query:{COLORS['ENDC']} {query}")
        print(f"{COLORS['GREEN']}Response:{COLORS['ENDC']} {response['response']}")
        print(f"{COLORS['YELLOW']}Context:{COLORS['ENDC']} {response['context'][:200]}...") 