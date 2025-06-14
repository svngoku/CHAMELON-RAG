import argparse
import time
from typing import Dict, Any, List, Optional
from setup import PipelineFactory
from chameleon.utils.logging_utils import setup_colored_logger, COLORS

def explain_rag_type(rag_type: str) -> str:
    """Provide educational explanation of each RAG type."""
    explanations = {
        "naive": """
🔍 NAIVE RAG:
- Simple retrieval + generation approach
- Basic similarity search without advanced filtering
- Good for: Simple Q&A, getting started with RAG
- Limitations: May retrieve irrelevant documents, no context optimization
        """,
        "modular": """
🔧 MODULAR RAG:
- Configurable components with better retrieval strategies
- Improved document filtering and ranking
- Good for: Production systems, customizable workflows
- Benefits: Better precision, configurable parameters
        """,
        "advanced": """
🚀 ADVANCED RAG:
- Multi-stage retrieval with reranking
- Context compression and query enhancement
- Good for: Complex queries, high-accuracy requirements
- Benefits: Best retrieval quality, handles complex reasoning
        """
    }
    return explanations.get(rag_type, "Unknown RAG type")

def demonstrate_rag_technique(
    rag_type: str, 
    documents: list, 
    config: Dict[str, Any] = None,
    custom_queries: Optional[List[str]] = None,
    show_details: bool = True
):
    """Demonstrate a specific RAG technique with comprehensive analysis."""
    logger = setup_colored_logger()
    
    # Show explanation
    if show_details:
        print(explain_rag_type(rag_type))
        print("-" * 60)
    
    try:
        # Create pipeline with specified RAG type
        start_time = time.time()
        pipeline = PipelineFactory.create_pipeline(
            documents=documents,
            rag_type=rag_type,
            memory_type="buffer",
            **config or {}
        )
        setup_time = time.time() - start_time
        
        logger.info(f"{COLORS['GREEN']}✅ {rag_type.upper()} pipeline created in {setup_time:.2f}s{COLORS['ENDC']}")
        
        # Use custom queries if provided, otherwise use defaults
        if custom_queries:
            queries = custom_queries
        else:
            # Enhanced example queries with different complexity levels
            queries = {
                "naive": [
                    "What are the basic characteristics of RAG systems?",
                    "How does RAG work in simple terms?",
                    "What is retrieval-augmented generation?"
                ],
                "modular": [
                    "Compare different RAG architectures and their trade-offs",
                    "What are the advantages of modular RAG systems over naive approaches?",
                    "How do you optimize RAG pipeline performance?"
                ],
                "advanced": [
                    "Analyze the evolution of RAG systems and their impact on AI applications",
                    "What advanced techniques improve RAG performance in complex domains?",
                    "How do multi-stage retrieval and reranking enhance answer quality?"
                ]
            }[rag_type]
        
        logger.info(f"{COLORS['BLUE']}🔄 Running {len(queries)} queries for {rag_type.upper()} RAG{COLORS['ENDC']}")
        
        results = []
        total_query_time = 0
        
        for i, query in enumerate(queries, 1):
            logger.info(f"{COLORS['BLUE']}Query {i}/{len(queries)}: {query[:50]}...{COLORS['ENDC']}")
            
            try:
                query_start = time.time()
                response = pipeline.run(query)
                query_time = time.time() - query_start
                total_query_time += query_time
                
                # Store results for analysis
                results.append({
                    'query': query,
                    'response': response,
                    'time': query_time
                })
                
                # Display results
                print(f"\n{COLORS['GREEN']}📝 Query {i}:{COLORS['ENDC']} {query}")
                print(f"{COLORS['GREEN']}⚡ Response Time:{COLORS['ENDC']} {query_time:.2f}s")
                print(f"{COLORS['GREEN']}🤖 Response:{COLORS['ENDC']} {response['response']}")
                
                if show_details and 'context' in response:
                    context_preview = response['context'][:300] + "..." if len(response['context']) > 300 else response['context']
                    print(f"{COLORS['YELLOW']}📄 Context Used:{COLORS['ENDC']} {context_preview}")
                
                if 'metadata' in response and show_details:
                    metadata = response['metadata']
                    if 'num_docs' in metadata:
                        print(f"{COLORS['CYAN']}📊 Documents Retrieved:{COLORS['ENDC']} {metadata['num_docs']}")
                
                print("-" * 80)
                
            except Exception as e:
                logger.error(f"{COLORS['RED']}❌ Error processing query {i}: {str(e)}{COLORS['ENDC']}")
                results.append({
                    'query': query,
                    'response': {'response': f"Error: {str(e)}"},
                    'time': 0,
                    'error': True
                })
                continue
        
        # Summary statistics
        successful_queries = [r for r in results if not r.get('error', False)]
        if successful_queries:
            avg_time = total_query_time / len(successful_queries)
            print(f"\n{COLORS['GREEN']}📈 {rag_type.upper()} RAG Summary:{COLORS['ENDC']}")
            print(f"  • Setup Time: {setup_time:.2f}s")
            print(f"  • Successful Queries: {len(successful_queries)}/{len(queries)}")
            print(f"  • Average Query Time: {avg_time:.2f}s")
            print(f"  • Total Query Time: {total_query_time:.2f}s")
        
        return results
            
    except Exception as e:
        logger.error(f"{COLORS['RED']}❌ Failed to create {rag_type} RAG pipeline: {str(e)}{COLORS['ENDC']}")
        if "api_key" in str(e).lower():
            logger.info(f"{COLORS['YELLOW']}💡 Tip: Set your OPENAI_API_KEY environment variable{COLORS['ENDC']}")
        return []

def interactive_mode(documents: list):
    """Allow users to interactively test queries."""
    logger = setup_colored_logger()
    
    print(f"\n{COLORS['BLUE']}🎯 Interactive RAG Testing Mode{COLORS['ENDC']}")
    print("Enter your queries (type 'quit' to exit, 'help' for commands)")
    
    # Let user choose RAG type
    print("\nAvailable RAG types:")
    print("1. naive - Simple retrieval + generation")
    print("2. modular - Configurable components")
    print("3. advanced - Multi-stage with reranking")
    
    while True:
        try:
            choice = input(f"\n{COLORS['GREEN']}Choose RAG type (1-3) or 'quit': {COLORS['ENDC']}").strip()
            if choice.lower() == 'quit':
                break
            
            rag_types = {'1': 'naive', '2': 'modular', '3': 'advanced'}
            if choice not in rag_types:
                print(f"{COLORS['RED']}Invalid choice. Please enter 1, 2, 3, or 'quit'{COLORS['ENDC']}")
                continue
            
            rag_type = rag_types[choice]
            
            # Get custom queries
            queries = []
            print(f"\n{COLORS['BLUE']}Enter queries for {rag_type} RAG (empty line to start):{COLORS['ENDC']}")
            while True:
                query = input("Query: ").strip()
                if not query:
                    break
                queries.append(query)
            
            if queries:
                demonstrate_rag_technique(rag_type, documents, custom_queries=queries, show_details=True)
            else:
                print(f"{COLORS['YELLOW']}No queries entered. Using default queries.{COLORS['ENDC']}")
                demonstrate_rag_technique(rag_type, documents, show_details=True)
                
        except KeyboardInterrupt:
            print(f"\n{COLORS['YELLOW']}Exiting interactive mode...{COLORS['ENDC']}")
            break
        except Exception as e:
            logger.error(f"{COLORS['RED']}Error in interactive mode: {str(e)}{COLORS['ENDC']}")

def compare_rag_types(documents: list, test_query: str = "What are advanced RAG techniques?"):
    """Compare all RAG types on the same query."""
    logger = setup_colored_logger()
    
    print(f"\n{COLORS['BLUE']}🔍 RAG Comparison Mode{COLORS['ENDC']}")
    print(f"Testing query: '{test_query}'")
    print("=" * 80)
    
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
                "filtering_threshold": 0.3
            },
            "generator_config": {
                "temperature": 0.5,
                "max_tokens": 8192,
                "model_name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                "provider": "together"
            }
        }
    }
    
    comparison_results = {}
    
    for rag_type, config in configs.items():
        print(f"\n{COLORS['CYAN']}Testing {rag_type.upper()} RAG:{COLORS['ENDC']}")
        results = demonstrate_rag_technique(
            rag_type, 
            documents, 
            config, 
            custom_queries=[test_query],
            show_details=False
        )
        if results:
            comparison_results[rag_type] = results[0]
    
    # Show comparison summary
    if comparison_results:
        print(f"\n{COLORS['GREEN']}📊 COMPARISON SUMMARY{COLORS['ENDC']}")
        print("=" * 80)
        for rag_type, result in comparison_results.items():
            response_length = len(result['response']['response'])
            print(f"{COLORS['BLUE']}{rag_type.upper()}:{COLORS['ENDC']}")
            print(f"  • Response Time: {result['time']:.2f}s")
            print(f"  • Response Length: {response_length} chars")
            print(f"  • Response Preview: {result['response']['response'][:100]}...")
            print()

def main():
    parser = argparse.ArgumentParser(description="Enhanced RAG Demonstration")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare all RAG types")
    parser.add_argument("--rag-type", choices=["naive", "modular", "advanced"], help="Run specific RAG type only")
    parser.add_argument("--query", "-q", help="Custom query for comparison mode")
    parser.add_argument("--no-details", action="store_true", help="Hide detailed output")
    
    args = parser.parse_args()
    
    logger = setup_colored_logger()
    
    try:
        # Load test documents
        logger.info(f"{COLORS['BLUE']}📚 Loading test documents...{COLORS['ENDC']}")
        documents = PipelineFactory.load_test_data_from_dataset()
        logger.info(f"{COLORS['GREEN']}✅ Loaded {len(documents)} documents{COLORS['ENDC']}")
        
        if args.interactive:
            interactive_mode(documents)
        elif args.compare:
            test_query = args.query or "What are advanced RAG techniques?"
            compare_rag_types(documents, test_query)
        elif args.rag_type:
            # Run specific RAG type
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
                        "filtering_threshold": 0.3
                    },
                    "generator_config": {
                        "temperature": 0.5,
                        "max_tokens": 500
                    }
                }
            }
            demonstrate_rag_technique(
                args.rag_type, 
                documents, 
                configs[args.rag_type],
                show_details=not args.no_details
            )
        else:
            # Default: demonstrate all RAG techniques
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
                        "filtering_threshold": 0.3
                    },
                    "generator_config": {
                        "temperature": 0.5,
                        "max_tokens": 500
                    }
                }
            }
            
            for rag_type, config in configs.items():
                demonstrate_rag_technique(
                    rag_type, 
                    documents, 
                    config,
                    show_details=not args.no_details
                )
                print("\n" + "="*80 + "\n")
                
    except Exception as e:
        logger.error(f"{COLORS['RED']}❌ Application error: {str(e)}{COLORS['ENDC']}")
        if "load_test_data_from_dataset" in str(e):
            logger.info(f"{COLORS['YELLOW']}💡 Tip: Make sure your setup.py has the load_test_data_from_dataset method{COLORS['ENDC']}")

if __name__ == "__main__":
    main() 