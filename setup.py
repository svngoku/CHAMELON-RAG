import unittest
from rag_techniques.pipeline.rag_pipeline import RAGPipeline
from rag_techniques.retrieval.simple_retriever import SimpleRetriever
from rag_techniques.vector_db_factory import VectorDBFactory
from rag_techniques.loaders import FileLoader, DocumentLoader
from rag_techniques.generation.llm_generator import LLMGenerator
from rag_techniques.preprocessing.markdown_chunking import MarkdownChunking
from rag_techniques.utils.logging_utils import setup_colored_logger, COLORS


def create_markdown_preprocessor():
    """Create and configure the markdown preprocessor."""
    return MarkdownChunking(
        chunk_size=500,
        chunk_overlap=50,
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
        return_each_line=False,
        strip_headers=True
    )


def create_generator():
    """Create and configure the LLM generator."""
    return LLMGenerator(
        provider="together",
        model="Qwen/Qwen2.5-Coder-32B-Instruct", 
        temperature=0.3,
        max_tokens=2000
    )


def create_retriever():
    """Create and configure the retriever with vector store."""
    vector_db_factory = VectorDBFactory(chunk_size=500, chunk_overlap=50)
    vectorstore = vector_db_factory.create_vectorstore(load_data(), store_type="faiss")
    return SimpleRetriever(vectorstore)


def create_pipeline():
    """Create and configure the RAG pipeline."""
    pipeline = RAGPipeline()
    pipeline.add_preprocessor(create_markdown_preprocessor())
    pipeline.set_retriever(create_retriever())
    pipeline.set_generator(create_generator())
    return pipeline


def load_data():
    """Load and return the dataset."""
    loader = DocumentLoader(
        chunk_size=1000,
        chunk_overlap=200,
        file_loader=FileLoader(supported_formats=[".txt", ".md"]),
    )
    file_paths = [
        "data/africanhistory1.txt",
        "data/africanhistory.txt"
    ]   
    return loader._load_multiple_files(file_paths)


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = create_pipeline()
        self.logger = setup_colored_logger()
        
    def test_pipeline_run(self):
        """Test basic pipeline functionality"""
        query = "Compare the emergence of Neolithic cultures in West Africa with those in the Nile Valley and Northern Horn of Africa ?"
        self.logger.info(f"{COLORS['BLUE']}Testing query: {query}{COLORS['ENDC']}")
        
        response = self.pipeline.run(query, load_data())
        
        self._print_results(query, response)
        self._verify_response(query, response)

    def _print_results(self, query, response):
        """Print test results in a formatted way."""
        print(f"\n{COLORS['GREEN']}=== Test Results ==={COLORS['ENDC']}")
        print(f"{COLORS['BLUE']}Query:{COLORS['ENDC']} {query}")
        print(f"{COLORS['GREEN']}Response:{COLORS['ENDC']} {response['response']}")
        print(f"{COLORS['YELLOW']}Context:{COLORS['ENDC']} {response['context'][:200]}...")

    def _verify_response(self, query, response):
        """Verify the response meets all requirements."""
        self.assertIsInstance(response, dict)
        self.assertIn('query', response)
        self.assertIn('context', response)
        self.assertIn('response', response)
        self.assertEqual(response['query'], query)
        self.assertIsInstance(response['response'], str)
        self.assertTrue(len(response['response']) > 0, "Response should not be empty")
        
    def test_pipeline_components(self):
        """Test if pipeline components are properly initialized"""
        self.assertIsNotNone(self.pipeline.preprocessors)
        self.assertIsNotNone(self.pipeline.retriever)
        self.assertIsNotNone(self.pipeline.generator)
        self.assertEqual(len(self.pipeline.preprocessors), 1)

    def test_markdown_preprocessing(self):
        """Test Markdown preprocessing"""
        markdown_text = """# Introduction
        
        ## Background
        This is some background information.
        More background details here.
        
        ## Methods
        Here are the methods we used.
        
        ### Data Collection
        Details about data collection.
        """
        
        preprocessor = MarkdownChunking()
        chunks = preprocessor.process(markdown_text)
        
        self.logger.info(f"{COLORS['GREEN']}Markdown Chunks:{COLORS['ENDC']}")
        for i, chunk in enumerate(chunks):
            self.logger.info(f"{COLORS['BLUE']}Chunk {i}:{COLORS['ENDC']}")
            self.logger.info(f"Content: {chunk.page_content[:100]}...")
            self.logger.info(f"Headers: {chunk.metadata}")


if __name__ == '__main__':
    unittest.main()
