from rag_techniques.pipeline.rag_pipeline import RAGPipeline
from rag_techniques.preprocessing.semantic_chunking import SemanticChunking
from rag_techniques.retrieval.fusion_retrieval import FusionRetrieval
from rag_techniques.postprocessing.intelligent_reranking import IntelligentReranking
from rag_techniques.generation.llm_generator import LLMGenerator


# Create and configure the RAG pipeline
pipeline = RAGPipeline()
pipeline.set_retriever(FusionRetrieval(chunk_size=1000, chunk_overlap=200, k=5, alpha=0.5))

# Process the data
data = [
    "RAG is a pipeline that combines retrieval and generation techniques to answer questions.",
    "Retrieval techniques help find relevant documents or passages.",
    "Generation techniques use language models to generate answers based on the retrieved information.",
    "RAG can improve the accuracy and efficiency of question-answering systems.",
    "However, it requires careful configuration and tuning to achieve optimal results.",
]  # Implement this function to load your dataset


# Process the data
pipeline.retriever.process(data)

# Now you can run queries
query = "What are the benefits of RAG?"
response = pipeline.run(query, data)

print(response)