from pydantic import BaseModel
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
import logging


class LLMGenerator(BaseGenerator, BaseModel):
    provider: str = "openai"
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    llm: Any = None
    output_parser: Any = None
    chain: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Initialize the LLM Generator."""
        super().__init__(**{
            'provider': provider.lower(),
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'chain': None
        })
        self.llm = self._initialize_llm(**kwargs)
        self.output_parser = StrOutputParser()
        
        # Initialize the chain with a prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based on the provided context.
        
        Context: {context}
        
        Question: {question}
        
        Answer:""")
        
        self.chain = prompt | self.llm | self.output_parser

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

    def generate(self, query: str, context: str) -> str:
        """Generate a response using the LLM."""
        try:
            inputs = {
                "context": context,
                "question": query
            }
            
            logging.info(f"Generating response for query: {query}")
            logging.info(f"Using context: {context[:200]}...")  # Log first 200 chars of context
            
            response = self.chain.invoke(inputs)
            
            logging.info(f"Generated response: {response}")
            return response
            
        except Exception as e:
            logging.error(f"Error in LLM generation: {str(e)}")
            raise

    def invoke(self, input_data: Dict[str, Any]) -> str:
        """Invoke method to make the generator compatible with LangChain's chain interface."""
        if isinstance(input_data, dict):
            return self.generate(
                input_data.get("prompt_template", "{question}"),
                input_data
            )
        return str(input_data)
