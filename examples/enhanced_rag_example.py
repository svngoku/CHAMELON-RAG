from chameleon.pipeline.enhanced_rag_pipeline import EnhancedRAGPipeline
from chameleon.base import PipelineConfig, RetrieverConfig, GeneratorConfig, MemoryConfig
from chameleon.preprocessing.query_transformer import QueryTransformer
from chameleon.postprocessing.contextual_compressor import ContextualCompressor
from chameleon.memory.entity_memory import EntityMemory
# from chameleon.evaluation.rag_evaluator import RAGEvaluator  # Temporarily disabled due to RAGAS compatibility
from chameleon.utils.logging_utils import setup_colored_logger, COLORS
from langchain_core.tools import Tool
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, Any, List
import os
import datetime

def load_documents(data_dir: str = "data"):
    """Load documents from the data directory."""
    logger = setup_colored_logger()
    logger.info(f"Loading documents from {data_dir}")
    
    try:
        # Load all text files from the data directory
        loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        
        logger.info(f"Loaded {len(documents)} documents and split into {len(split_docs)} chunks")
        return split_docs
    
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise

def create_search_tool():
    """Create a simple search tool for demonstration."""
    def search_web(query: str) -> str:
        """Search the web for information (simulated)."""
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        return f"Simulated web search results for '{query}' as of {current_date}:\n" + \
               "- Found relevant information from multiple sources\n" + \
               "- Latest research indicates significant progress in this area\n" + \
               "- Several recent developments might be relevant to your query"
    
    return Tool(
        name="web_search",
        description="Search the web for recent or additional information not in the documents",
        func=search_web
    )

def demonstrate_enhanced_rag(documents: List, config: Dict[str, Any] = None):
    """Demonstrate the enhanced RAG pipeline with various features."""
    logger = setup_colored_logger()
    
    try:
        # Create default config if none provided
        if not config:
            config = {
                "retriever_config": {
                    "top_k": 5,
                    "similarity_threshold": 0.3,
                    "retrieval_type": "semantic",
                    "reranking_enabled": True,
                    "filtering_enabled": True,
                    "filtering_threshold": 0.2,
                    "embedding_model": "all-MiniLM-L6-v2",
                    "multi_query_enabled": True,
                    "parent_document_enabled": False
                },
                "generator_config": {
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "model_name": "gpt-3.5-turbo"
                },
                "memory_config": {
                    "memory_type": "entity",
                    "max_history": 10
                }
            }
        
        # Create pipeline config
        pipeline_config = PipelineConfig(
            rag_type="enhanced",
            retriever_config=RetrieverConfig(**config["retriever_config"]),
            generator_config=GeneratorConfig(**config["generator_config"]),
            memory_config=MemoryConfig(**config["memory_config"]),
            chain_type="stuff"
        )
        
        # Create preprocessors (QueryTransformer should not be used for document preprocessing)
        # query_transformer = QueryTransformer(technique="expansion")
        
        # Create postprocessors
        contextual_compressor = ContextualCompressor(
            compression_mode="paragraph",
            min_relevance_score=0.7
        )
        
        # Create tools
        tools = [create_search_tool()]
        
        # Create enhanced RAG pipeline
        pipeline = EnhancedRAGPipeline(
            title="Enhanced RAG Demo",
            documents=documents,
            config=pipeline_config,
            preprocessors=[],  # No document preprocessors for now
            postprocessors=[contextual_compressor],
            tools=tools,
            enable_evaluation=False  # Disabled due to RAGAS compatibility issues
        )
        
        logger.info(f"{COLORS['BLUE']}Enhanced RAG Pipeline initialized{COLORS['ENDC']}")
        
        # Example queries to demonstrate different features
        queries = [
            "What are advanced RAG techniques?",
            "Tell me more about parent document retrieval",
            "How does entity memory improve conversations?",
            "What are the latest developments in contextual compression?",
            "Compare different retrieval methods used in RAG"
        ]
        
        for query in queries:
            logger.info(f"{COLORS['BLUE']}Processing query: {query}{COLORS['ENDC']}")
            
            # Run the pipeline
            response = pipeline.run(query)
            
            # Display results
            print(f"\n{COLORS['GREEN']}Query:{COLORS['ENDC']} {query}")
            
            if "transformed_query" in response:
                print(f"{COLORS['YELLOW']}Transformed Query:{COLORS['ENDC']} {response['transformed_query']}")
            
            if "tool_result" in response and response["tool_result"].get("tool_used"):
                print(f"{COLORS['YELLOW']}Tool Used:{COLORS['ENDC']} {response['tool_result']['tool_used']}")
                print(f"{COLORS['YELLOW']}Tool Result:{COLORS['ENDC']} {response['tool_result']['tool_result']}")
            
            print(f"{COLORS['GREEN']}Response:{COLORS['ENDC']} {response['response']}")
            
            # Note: Evaluation temporarily disabled due to RAGAS compatibility issues
            # if "evaluation" in response:
            #     eval_data = response["evaluation"]
            #     overall_score = eval_data.get("overall_score", 0)
            #     print(f"{COLORS['YELLOW']}Evaluation Score:{COLORS['ENDC']} {overall_score:.2f}/1.0")
            #     
            #     if eval_data.get("hallucination_detected"):
            #         print(f"{COLORS['RED']}⚠️ Potential hallucination detected!{COLORS['ENDC']}")
            
            print("-" * 80)
        
        # Get pipeline statistics
        stats = pipeline.get_stats()
        print(f"\n{COLORS['BLUE']}Pipeline Statistics:{COLORS['ENDC']}")
        for key, value in stats.items():
            print(f"- {key}: {value}")
        
    except Exception as e:
        logger.error(f"{COLORS['RED']}Error in enhanced RAG pipeline: {str(e)}{COLORS['ENDC']}")
        raise

def main():
    # Load documents
    documents = load_documents()
    
    # Demonstrate enhanced RAG pipeline
    demonstrate_enhanced_rag(documents)

if __name__ == "__main__":
    main()