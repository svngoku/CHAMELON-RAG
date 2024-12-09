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

class LLMGenerator(BaseGenerator):
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        self._llm = self._initialize_llm()
        self._output_parser = StrOutputParser()
    
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
        
        if self.config.provider not in providers:
            raise ValueError(f"Unsupported provider: {self.config.provider}. "
                           f"Choose from {list(providers.keys())}")
        
        llm_class = providers[self.config.provider]
        llm_kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "model": self.config.model_name
        }
            
        return llm_class(**llm_kwargs)

    def generate(self, query: str, context: List[Document]) -> str:
        """Generate response from context."""
        # Convert documents to a single context string
        context_str = "\n\n".join(doc.page_content for doc in context)
        
        # Truncate context if needed
        max_context_chars = 80000  # Approximately 20k tokens
        if len(context_str) > max_context_chars:
            context_str = context_str[:max_context_chars] + "..."
        
        prompt_template = """
        Based on the following context, please answer the question.
        
        Context: {context}
        
        Question: {query}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self._llm | self._output_parser
        
        return chain.invoke({
            "context": context_str,
            "query": query
        })

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
