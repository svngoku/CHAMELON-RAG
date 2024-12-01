from rag_techniques.base import BaseGenerator
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from typing import Dict, Any, Optional


class LLMGenerator(BaseGenerator):
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ):
        """Initialize the LLM Generator.
        
        Args:
            provider: The LLM provider to use ('openai', 'groq', 'mistral', 'cohere')
            model: The specific model to use (if None, uses provider's default)
            temperature: Controls randomness in the output (0.0 to 1.0)
            **kwargs: Additional arguments to pass to the LLM
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        
        # Initialize the appropriate LLM based on provider
        self.llm = self._initialize_llm(**kwargs)
        self.output_parser = StrOutputParser()

    def _initialize_llm(self, **kwargs) -> Any:
        """Initialize the specific LLM based on the provider."""
        providers = {
            "openai": ChatOpenAI,
            "groq": ChatGroq,
            "mistral": ChatMistralAI,
            "cohere": ChatCohere
        }
        
        if self.provider not in providers:
            raise ValueError(f"Unsupported provider: {self.provider}. "
                           f"Choose from {list(providers.keys())}")
        
        llm_class = providers[self.provider]
        llm_kwargs = {
            "temperature": self.temperature,
            **kwargs
        }
        
        if self.model:
            llm_kwargs["model"] = self.model
            
        return llm_class(**llm_kwargs)

    def generate(self, prompt_template: str, inputs: Dict[str, Any]) -> str:
        """Generate text using the configured LLM.
        
        Args:
            prompt_template: The template string for the prompt
            inputs: Dictionary of input variables for the prompt template
            
        Returns:
            Generated text string
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | self.output_parser
        return chain.invoke(inputs)
