from typing import List, Optional, Union, Dict, Any
from rag_techniques.base import BasePreprocessor, BaseRetriever, BaseGenerator, BaseMemory
from langchain_core.documents import Document
import logging
from rag_techniques.utils.logging_utils import COLORS

class RAGPipeline:
    def __init__(
        self,
        title: str,
        preprocessors: List[BasePreprocessor],
        retriever: BaseRetriever,
        generator: BaseGenerator,
        memory: Optional[BaseMemory] = None
    ):
        self.title = title
        self.preprocessors = preprocessors
        self.retriever = retriever
        self.generator = generator
        self.memory = memory
    
    def run(self, query: str, input_data: Union[str, List[Document]]) -> Dict[str, Any]:
        """Run the RAG pipeline on the input data."""
        try:
            # Get chat history if memory exists
            chat_history = ""
            if self.memory:
                chat_history = self.memory.load_memory_variables({}).get("chat_history", "")
                logging.info(f"{COLORS['BLUE']}Loaded chat history from memory{COLORS['ENDC']}")

            # Preprocess the input data
            processed_data = input_data
            for preprocessor in self.preprocessors:
                processed_data = preprocessor.process(processed_data)

            # Retrieve relevant documents
            relevant_docs = self.retriever.retrieve(query)
            context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
            
            logging.info(f"\n{COLORS['YELLOW']}Query:{COLORS['ENDC']} {query}")
            logging.info(f"\n{COLORS['YELLOW']}Context:{COLORS['ENDC']} {context_text[:200]}...")

            # Generate response using the context and chat history
            generated_response = self.generator.process(
                context=context_text,
                query=query,
                chat_history=chat_history
            )
            
            logging.info(f"\n{COLORS['GREEN']}Response:{COLORS['ENDC']} {generated_response}")

            # Save to memory if it exists
            if self.memory:
                self.memory.save_context(
                    {"input": query},
                    {"output": generated_response}
                )
                logging.info(f"{COLORS['BLUE']}Saved conversation to memory{COLORS['ENDC']}")

            return {
                "query": query,
                "context": context_text,
                "response": generated_response,
                "chat_history": chat_history
            }
            
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error in pipeline execution: {str(e)}{COLORS['ENDC']}")
            raise

    def to_ascii(self) -> str:
        """Generate an ASCII representation of the RAG pipeline components.
        
        Returns:
            str: A formatted string showing the pipeline components and their types
        """
        logging.info(f"{COLORS['BLUE']}Generating ASCII representation of the pipeline{COLORS['ENDC']}")
        
        # Format preprocessors section
        preprocessors_header = f"{COLORS['GREEN']}=== Preprocessors ==={COLORS['ENDC']}"
        preprocessors_str = "\n".join(
            f"- {preprocessor.name}: {preprocessor.__class__.__name__}" 
            for preprocessor in self.preprocessors
        )
        
        # Format retriever section  
        retriever_header = f"\n{COLORS['YELLOW']}=== Retriever ==={COLORS['ENDC']}"
        retriever_str = f"- {self.retriever.__class__.__name__}"
        
        # Format generator section
        generator_header = f"\n{COLORS['BLUE']}=== Generator ==={COLORS['ENDC']}"
        generator_str = f"- {self.generator.__class__.__name__}"
        
        return f"{preprocessors_header}\n{preprocessors_str}\n{retriever_header}\n{retriever_str}\n{generator_header}\n{generator_str}"