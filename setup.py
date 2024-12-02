import logging
import unittest
from rag_techniques.pipeline.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from rag_techniques.retrieval.simple_retriever import SimpleRetriever
from rag_techniques.vector_db_factory import VectorDBFactory
from rag_techniques.loaders import FileLoader
from rag_techniques.generation.simple_generator import SimpleGenerator
from rag_techniques.generation.llm_generator import LLMGenerator
from rag_techniques.utils.utils import encode_pdf, encode_from_string, read_pdf_to_string
from rag_techniques.preprocessing.semantic_chunking import SemanticChunking
from langchain.prompts import ChatPromptTemplate

def create_pipeline():
    """Create and configure the RAG pipeline."""
    # Initialize pipeline components
    preprocessor = SemanticChunking()
    vector_db_factory = VectorDBFactory(chunk_size=1000, chunk_overlap=200)
    vectorstore = vector_db_factory.create_vectorstore(load_data(), store_type="faiss")
    retriever = SimpleRetriever(vectorstore)
    generator = LLMGenerator(provider="together", model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", temperature=0.7, max_tokens=2000)

    # Configure pipeline
    pipeline = RAGPipeline()
    pipeline.add_preprocessor(preprocessor)
    pipeline.set_retriever(retriever) 
    pipeline.set_generator(generator)

    return pipeline

def load_data():
    """Load and return the dataset."""
    loader = FileLoader()
    file_paths = [
        "data/africanhistory1.txt",
        "data/africanhistory.txt"
    ]   
    return loader.load_files(file_paths)

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = create_pipeline()
        
    def test_pipeline_run(self):
        """Test basic pipeline functionality"""
        query = "What were the major Neolithic cultures in West Africa and when did they exist?"
        data = load_data()
        
        # Test pipeline execution
        response = self.pipeline.run(query, data)
        
        # Print the response for visibility
        print("\nTest Results:")
        print(f"Query: {query}")
        print(f"Response: {response['response']}")
        print(f"Context used: {response['context'][:200]}...")  # First 200 chars
        
        # Basic assertions
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

if __name__ == '__main__':
    unittest.main()
