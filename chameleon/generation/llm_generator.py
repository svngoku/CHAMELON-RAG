from chameleon.base import BaseGenerator
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_together import ChatTogether
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from typing import Dict, Any, List
from chameleon.base import GeneratorConfig
from langchain.schema import Document
import logging
from chameleon.utils.logging_utils import COLORS

class LLMGenerator(BaseGenerator):
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        self.provider = config.provider
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        logging.info(f"{COLORS['BLUE']}Initializing {self.provider} LLM generator with model {self.model_name}{COLORS['ENDC']}")
        self._llm = self._initialize_llm()
        self._output_parser = StrOutputParser()
        
        # Create the prompt template once during initialization
        self._prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following context to answer the question."),
            ("user", """Context: {context}
            
            Chat History: {chat_history}
            
            Question: {query}""")
        ])
    
    def _initialize_llm(self) -> Any:
        """Initialize the specific LLM based on the provider."""
        providers = {
            "openai": ChatOpenAI,
            "groq": ChatGroq,
            "mistral": ChatMistralAI,
            "cohere": ChatCohere,
            "together": ChatTogether,
            "vertexai": ChatGoogleGenerativeAI
        }
        
        if self.provider not in providers:
            error_msg = f"Unsupported provider: {self.provider}. Choose from {list(providers.keys())}"
            logging.error(f"{COLORS['RED']}{error_msg}{COLORS['ENDC']}")
            raise ValueError(error_msg)
        
        logging.info(f"{COLORS['BLUE']}Configuring {self.provider} LLM with temperature={self.temperature}, max_tokens={self.max_tokens}{COLORS['ENDC']}")
        
        llm_class = providers[self.provider]
        llm_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model": self.model_name
        }
            
        return llm_class(**llm_kwargs)
    
    def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Generate a response using the LLM."""
        logging.info(f"{COLORS['BLUE']}Generating response for query: {query[:100]}...{COLORS['ENDC']}")
        
        context = "\n\n".join(doc.page_content for doc in documents)
        chat_history = ""  # You can add chat history handling here if needed
        
        logging.info(f"{COLORS['BLUE']}Using {len(documents)} documents for context{COLORS['ENDC']}")
        
        try:
            # Create the chain using the prompt template
            chain = self._prompt_template | self._llm | self._output_parser
            
            # Invoke the chain with the inputs
            response = chain.invoke({
                "context": context, 
                "query": query,
                "chat_history": chat_history
            })
            
            logging.info(f"{COLORS['GREEN']}Successfully generated response{COLORS['ENDC']}")
            
            return {
                'response': response,
                'metadata': {
                    'model': self.model_name,
                    'provider': self.provider,
                    'temperature': self.temperature
                }
            }
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logging.error(f"{COLORS['RED']}{error_msg}{COLORS['ENDC']}")
            raise

    def process(self, context: str, query: str, chat_history: str = "") -> str:
        """Process the context and query to generate a response."""
        # Truncate context to roughly 20k tokens (approximate using characters)
        max_context_chars = 80000  # Approximately 20k tokens
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "..."
        
        prompt_template = """
        Based on the following context and chat history, please answer the question.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Question: {query}
        
        Answer:"""
        
        return self.generate(prompt_template, {
            "context": context, 
            "query": query,
            "chat_history": chat_history
        })
