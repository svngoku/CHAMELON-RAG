import logging
import unittest
from rag_techniques.pipeline.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from rag_techniques.retrieval.simple_retriever import SimpleRetriever
from rag_techniques.vector_db_factory import VectorDBFactory
from rag_techniques.loaders import FileLoader
from rag_techniques.generation.simple_generator import SimpleGenerator
from rag_techniques.utils.utils import encode_pdf, encode_from_string, read_pdf_to_string
from rag_techniques.preprocessing.semantic_chunking import SemanticChunking

def create_pipeline():
    """Create and configure the RAG pipeline."""
    # Initialize pipeline components
    preprocessor = SemanticChunking()
    vector_db_factory = VectorDBFactory(chunk_size=1000, chunk_overlap=200)
    vectorstore = vector_db_factory.create_vectorstore(load_data(), store_type="faiss")
    retriever = SimpleRetriever(vectorstore)
    generator = SimpleGenerator()

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
        query = "What was the event that happened in West Africa in the Neolithic period?"
        data = ""
        
        # Test pipeline execution
        response = self.pipeline.run(query, data)
        
        # Basic assertions
        self.assertIsInstance(response, dict)
        self.assertIn('query', response)
        self.assertIn('context', response)
        self.assertIn('response', response)
        self.assertEqual(response['query'], query)
        
    def test_pipeline_components(self):
        """Test if pipeline components are properly initialized"""
        self.assertIsNotNone(self.pipeline._components.preprocessors)
        self.assertIsNotNone(self.pipeline._components.retriever)
        self.assertIsNotNone(self.pipeline._components.generator)
        self.assertEqual(len(self.pipeline._components.preprocessors), 1)

if __name__ == '__main__':
    unittest.main()
