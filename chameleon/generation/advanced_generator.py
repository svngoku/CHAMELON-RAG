from typing import List, Any, Dict, Optional, Generator
from ..base import BaseGenerator, GeneratorConfig
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
import json
import asyncio
from dataclasses import asdict

class AdvancedGenerator(BaseGenerator):
    """Advanced generator with enhanced generation capabilities and streaming support."""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        self.llm = self._create_llm()
        self.default_system_prompt = """You are a helpful AI assistant that provides accurate, 
        informative responses based on the given context. Always maintain a professional tone 
        and cite specific information from the context when possible."""
    
    def validate_config(self, config: GeneratorConfig) -> bool:
        """Validate generator configuration."""
        if config.temperature < 0 or config.temperature > 2:
            return False
        if config.max_tokens < 1:
            return False
        if config.generation_type not in ["chat", "completion", "stream"]:
            return False
        return True
    
    def _create_llm(self) -> Any:
        """Create LLM based on provider and configuration."""
        if self.config.provider == "openai":
            return ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.streaming
            )
        # Add support for other providers here
        raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def generate(self, query: str, context: List[Document], **kwargs) -> Dict[str, Any]:
        """Generate enhanced response using context."""
        # Prepare context string
        context_str = self._prepare_context(context)
        
        # Choose generation method
        if self.config.generation_type == "chat":
            response = self._generate_chat_response(query, context_str, **kwargs)
        elif self.config.generation_type == "completion":
            response = self._generate_completion_response(query, context_str, **kwargs)
        else:
            raise ValueError(f"Unsupported generation type: {self.config.generation_type}")
        
        # Post-process and prepare result
        processed_response = self._post_process_response(response)
        
        return {
            'response': processed_response,
            'metadata': {
                'model': self.config.model_name,
                'generation_type': self.config.generation_type,
                'context_length': len(context),
                'config': self.config.model_dump()
            }
        }
    
    async def stream(self, query: str, context: List[Document], **kwargs) -> Generator[str, None, None]: # type: ignore
        """Stream response generation."""
        if not self.config.streaming:
            raise ValueError("Streaming is not enabled in the configuration")
        
        context_str = self._prepare_context(context)
        
        async for chunk in self._stream_response(query, context_str, **kwargs):
            yield chunk
    
    def _generate_chat_response(self, query: str, context: str, **kwargs) -> str:
        """Generate response using chat format."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt or self.default_system_prompt),
            ("human", "Context:\n{context}\n\nQuery: {query}\n\nProvide a detailed response:"),
        ])
        
        chain = (
            {"context": RunnablePassthrough(), "query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke({"context": context, "query": query})
    
    def _generate_completion_response(self, query: str, context: str, **kwargs) -> str:
        """Generate response using completion format."""
        prompt = PromptTemplate(
            template="""Context: {context}\n\nQuery: {query}\n\nResponse:""",
            input_variables=["context", "query"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(context=context, query=query)
        
        return result
    
    async def _stream_response(self, query: str, context: str, **kwargs) -> Generator[str, None, None]: # type: ignore
        """Stream response generation."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt or self.default_system_prompt),
            ("human", "Context:\n{context}\n\nQuery: {query}\n\nProvide a detailed response:"),
        ])
        
        chain = (
            {"context": RunnablePassthrough(), "query": RunnablePassthrough()}
            | prompt
            | self.llm
        )
        
        async for chunk in chain.astream({"context": context, "query": query}):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context string from documents with metadata."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Extract metadata
            metadata = self._format_metadata(doc.metadata)
            # Add document content with metadata
            context_parts.append(f"[Document {i}]\nMetadata: {metadata}\nContent: {doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format document metadata."""
        if not metadata:
            return "No metadata available"
        
        # Format key metadata fields
        formatted = []
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            formatted.append(f"{key}: {value}")
        
        return "; ".join(formatted)
    
    def _post_process_response(self, response: str) -> str:
        """Post-process the generated response."""
        # Add citation markers if they don't exist
        if "[Document" not in response and len(response) > 100:
            response += "\n\nNote: This response is based on the provided context documents."
        
        return response.strip() 