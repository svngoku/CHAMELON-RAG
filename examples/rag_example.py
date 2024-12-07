from setup import PipelineFactory
from rag_techniques.utils.logging_utils import setup_colored_logger, COLORS


def main():
    # Setup logging
    logger = setup_colored_logger()
    
    try:
        # Load documents
        documents = PipelineFactory.load_test_data_from_dataset()
        
        # Create pipeline with Redis memory for production
        pipeline = PipelineFactory.create_pipeline(
            documents=documents,
            memory_type="buffer"
        )
        
        # Example conversation
        queries = [
            "How does this relate to modern developments?"
        ]
        
        for query in queries:
            logger.info(f"{COLORS['BLUE']}Processing query: {query}{COLORS['ENDC']}")
            
            response = pipeline.run(query, documents)
            
            print(f"\n{COLORS['GREEN']}Query:{COLORS['ENDC']} {query}")
            print(f"{COLORS['GREEN']}Response:{COLORS['ENDC']} {response['response']}")
            print(f"{COLORS['YELLOW']}Chat History:{COLORS['ENDC']} Available")
            print("-" * 80)
            
    except Exception as e:
        logger.error(f"{COLORS['RED']}Error in RAG pipeline: {str(e)}{COLORS['ENDC']}")
        raise

if __name__ == "__main__":
    main() 