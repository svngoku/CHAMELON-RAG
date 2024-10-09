from rag_techniques.pipeline.rag_pipeline import RAGPipeline
from rag_techniques.retrieval.fusion_retrieval import FusionRetrieval

def create_pipeline():
    """Create and configure the RAG pipeline."""
    pipeline = RAGPipeline()
    pipeline.set_retriever(FusionRetrieval(chunk_size=1000, chunk_overlap=200, k=5, alpha=0.5))
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
    pipeline.retriever.process(data)
    query = "What are the benefits of RAG?"
    response = pipeline.run(query, data)
    print(response)

if __name__ == "__main__":
    main()
