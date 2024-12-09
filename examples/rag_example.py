from setup import PipelineFactory
from chameleon.utils.logging_utils import setup_colored_logger, COLORS
from typing import Dict, Any

def demonstrate_rag_technique(rag_type: str, documents: list, config: Dict[str, Any] = None):
    """Demonstrate a specific RAG technique with example queries."""
    logger = setup_colored_logger()
    
    try:
        # Create pipeline with specified RAG type
        pipeline = PipelineFactory.create_pipeline(
            documents=documents,
            rag_type=rag_type,
            memory_type="buffer",
            **config or {}
        )
        
        # Example queries tailored to RAG type
        queries = {
            "naive": [
                "What are the basic characteristics of RAG systems?",
                "How does RAG work in simple terms?"
            ],
            "modular": [
                "Compare different RAG architectures",
                "What are the advantages of modular RAG systems?"
            ],
            "advanced": [
                "Analyze the evolution of RAG systems and their impact",
                "What advanced techniques improve RAG performance?"
            ]
        }
        
        logger.info(f"{COLORS['BLUE']}Demonstrating {rag_type.upper()} RAG Pipeline{COLORS['ENDC']}")
        
        for query in queries[rag_type]:
            logger.info(f"{COLORS['BLUE']}Processing query: {query}{COLORS['ENDC']}")
            
            response = pipeline.run(query, documents)
            
            print(f"\n{COLORS['GREEN']}Query:{COLORS['ENDC']} {query}")
            print(f"{COLORS['GREEN']}Response:{COLORS['ENDC']} {response['response']}")
            if 'context' in response:
                print(f"{COLORS['YELLOW']}Context:{COLORS['ENDC']} {response['context'][:200]}...")
            print("-" * 80)
            
    except Exception as e:
        logger.error(f"{COLORS['RED']}Error in {rag_type} RAG pipeline: {str(e)}{COLORS['ENDC']}")
        raise

def main():
    # Load test documents
    documents = PipelineFactory.load_test_data_from_dataset()
    
    # Demonstrate each RAG technique
    configs = {
        "naive": {},
        "modular": {
            "retriever_config": {"top_k": 3},
            "generator_config": {"temperature": 0.7}
        },
        "advanced": {
            "retriever_config": {
                "top_k": 5,
                "reranking_enabled": True,
                "filtering_threshold": 0.8
            },
            "generator_config": {
                "temperature": 0.5,
                "max_tokens": 500
            }
        }
    }
    
    for rag_type, config in configs.items():
        demonstrate_rag_technique(rag_type, documents, config)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 