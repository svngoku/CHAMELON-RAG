from rag_techniques.pipeline.rag_pipeline import RAGPipeline
from rag_techniques.retrieval.simple_retriever import SimpleRetriever
from rag_techniques.vector_db_factory import VectorDBFactory
from rag_techniques.generation.simple_generator import SimpleGenerator

def create_pipeline():
    """Create and configure the RAG pipeline."""
    pipeline = RAGPipeline()
    vector_db_factory = VectorDBFactory(chunk_size=1000, chunk_overlap=200)
    vectorstore = vector_db_factory.create_vectorstore(load_data())
    pipeline.set_retriever(SimpleRetriever(vectorstore))
    pipeline.generator = SimpleGenerator()
    return pipeline

def load_data():
    """Load and return the dataset."""
    return [
        "RAG is a pipeline that combines retrieval and generation techniques to answer questions.",
        "Retrieval techniques help find relevant documents or passages.",
        "Generation techniques use language models to generate answers based on the retrieved information.",
        "RAG can improve the accuracy and efficiency of question-answering systems.",
        "However, it requires careful configuration and tuning to achieve optimal results.",
    ]

def main():
    pipeline = create_pipeline()
    data = load_data()
    query = "What are the benefits of RAG?"
    response = pipeline.run(query, data)
    print(response)

if __name__ == "__main__":
    main()
