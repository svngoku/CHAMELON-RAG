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

# Import LiteLLM for additional provider support
try:
    from langchain_community.llms import LiteLLM
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

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
            ("system", """You are an expert AI assistant specializing in providing accurate, contextual answers. 
            
            IMPORTANT INSTRUCTIONS:
            - Base your answer primarily on the provided context
            - If the context doesn't contain enough information, clearly state what's missing
            - Be specific and cite relevant details from the context
            - Avoid generic responses - focus on the specific information provided
            - If asked about comparisons or technical details, use the context to provide concrete examples"""),
            ("user", """Context: {context}
            
            Chat History: {chat_history}
            
            Question: {query}
            
            Please provide a detailed answer based on the context above.""")
        ])
    
    def _initialize_llm(self) -> Any:
        """Initialize the specific LLM based on the provider."""
        providers = {
            "openai": self._create_openai_llm,
            "groq": self._create_groq_llm,
            "mistral": self._create_mistral_llm,
            "cohere": self._create_cohere_llm,
            "together": self._create_together_llm,
            "vertexai": self._create_vertexai_llm,
            "litellm": self._create_litellm_llm,
            "openrouter": self._create_openrouter_llm
        }
        
        if self.provider not in providers:
            error_msg = f"Unsupported provider: {self.provider}. Choose from {list(providers.keys())}"
            logging.error(f"{COLORS['RED']}{error_msg}{COLORS['ENDC']}")
            raise ValueError(error_msg)
        
        logging.info(f"{COLORS['BLUE']}Configuring {self.provider} LLM with temperature={self.temperature}, max_tokens={self.max_tokens}{COLORS['ENDC']}")
        
        return providers[self.provider]()
    
    def _create_openai_llm(self):
        """Create OpenAI LLM."""
        return ChatOpenAI(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model_name
        )
    
    def _create_groq_llm(self):
        """Create Groq LLM."""
        return ChatGroq(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model_name
        )
    
    def _create_mistral_llm(self):
        """Create Mistral LLM."""
        return ChatMistralAI(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model_name
        )
    
    def _create_cohere_llm(self):
        """Create Cohere LLM."""
        return ChatCohere(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model_name
        )
    
    def _create_together_llm(self):
        """Create Together AI LLM."""
        return ChatTogether(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model_name
        )
    
    def _create_vertexai_llm(self):
        """Create Google Vertex AI LLM."""
        return ChatGoogleGenerativeAI(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            model=self.model_name
        )
    
    def _create_litellm_llm(self):
        """Create LiteLLM instance for multiple provider support."""
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM not available. Install with: pip install litellm")
        
        # LiteLLM supports many providers through a unified interface
        # Model name should include provider prefix (e.g., "anthropic/claude-3-sonnet", "openai/gpt-4")
        return LiteLLM(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def _create_openrouter_llm(self):
        """Create OpenRouter LLM using OpenAI-compatible interface."""
        import os
        
        # Create ChatOpenAI instance with OpenRouter configuration
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/chameleon-rag",
                "X-Title": "CHAMELEON RAG Framework"
            }
        )
        
        return llm
    
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
