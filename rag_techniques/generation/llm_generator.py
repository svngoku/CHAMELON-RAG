from rag_techniques.base import BaseGenerator
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_together import ChatTogether
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from typing import Dict, Any, Optional
from pydantic import Field, PrivateAttr
from rag_techniques.base import BaseMemory

class LLMGenerator(BaseGenerator):
    provider: str = Field(default="openai", description="The LLM provider to use")
    model: Optional[str] = Field(default=None, description="The specific model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Controls randomness in output")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens in output")
    memory: Optional[BaseMemory] = Field(default=None, description="Memory component to use")
    
    # Private attributes using Pydantic's PrivateAttr
    _llm: Any = PrivateAttr(default=None)
    _output_parser: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any = None) -> None:
        """Initialize after Pydantic validation.
        
        Args:
            context: Additional context from Pydantic (unused but required)
        """
        super().model_post_init(context)
        self._llm = self._initialize_llm()
        self._output_parser = StrOutputParser()
        self.memory = self.memory

    def _initialize_llm(self, **kwargs) -> Any:
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
            raise ValueError(f"Unsupported provider: {self.provider}. "
                           f"Choose from {list(providers.keys())}")
        
        llm_class = providers[self.provider]
        llm_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs
        }
        
        if self.model:
            llm_kwargs["model"] = self.model
            
        return llm_class(**llm_kwargs)

    def generate(self, prompt_template: str, inputs: Dict[str, Any]) -> str:
        """Generate text using the configured LLM."""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self._llm | self._output_parser
        return chain.invoke(inputs)

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
