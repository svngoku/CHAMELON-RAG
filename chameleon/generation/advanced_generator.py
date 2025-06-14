from typing import List, Dict, Any, Optional, Generator, Union
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dataclasses import asdict
import time

from ..base import BaseGenerator, GeneratorConfig


class AdvancedGenerator(BaseGenerator):
    """Advanced generator with support for multiple LLM providers and streaming."""
    
    def __init__(self, config: GeneratorConfig):
        """Initialize the advanced generator."""
        super().__init__(config)
        self.llm = self._create_llm()
        
        # Default system prompt for RAG
        self.default_system_prompt = """You are a helpful AI assistant that provides accurate and comprehensive responses based on the given context.
        
        Guidelines:
        - Answer the user's question based ONLY on the provided context
        - If the context doesn't contain enough information, acknowledge the limitations
        - Never make up information that's not in the context
        - Cite sources from the context when relevant
        - Format your response in a clear and readable way
        - If answering a coding question, explain the solution thoroughly
        """
    
    def validate_config(self, config: GeneratorConfig) -> bool:
        """Validate generator configuration."""
        return True  # All config values have appropriate defaults
    
    def generate(
        self, 
        query: str, 
        context: List[Document], 
        chat_history: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response based on the query and context.
        
        Args:
            query: User query
            context: Retrieved documents
            chat_history: Optional conversation history
            additional_context: Optional additional context (e.g., entity information)
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        # Prepare context string from documents
        context_str = self._prepare_context(context)
        
        # Add additional context if provided
        if additional_context:
            context_str = f"{additional_context}\n\n{context_str}"
        
        # Format history if provided
        history_str = ""
        if chat_history and len(chat_history) > 0:
            history_items = []
            for item in chat_history[-self.config.max_history_items:] if hasattr(self.config, 'max_history_items') else chat_history:
                if "query" in item and "response" in item:
                    history_items.append(f"User: {item['query']}")
                    history_items.append(f"Assistant: {item['response']}")
            history_str = "\n".join(history_items)
        
        # Generate based on generation type
        if self.config.generation_type == "chat":
            response = self._generate_chat_response(query, context_str, history_str)
        else:
            response = self._generate_completion_response(query, context_str, history_str)
        
        # Post-process response
        response = self._post_process_response(response)
        
        # Return response with metadata
        return {
            'response': response,
            'metadata': {
                'generation_time': time.time() - start_time,
                'model': self.config.model_name,
                'provider': self.config.provider,
                'num_context_docs': len(context),
                'context_length': len(context_str)
            }
        }
    
    async def stream(
        self, 
        query: str, 
        context: List[Document],
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Generator[str, None, None]:
        """Stream response generation."""
        if not self.config.streaming:
            raise ValueError("Streaming is not enabled in the configuration")
        
        # Prepare context
        context_str = self._prepare_context(context)
        
        # Format history if provided
        history_str = ""
        if chat_history and len(chat_history) > 0:
            history_items = []
            for item in chat_history[-self.config.max_history_items:] if hasattr(self.config, 'max_history_items') else chat_history:
                if "query" in item and "response" in item:
                    history_items.append(f"User: {item['query']}")
                    history_items.append(f"Assistant: {item['response']}")
            history_str = "\n".join(history_items)
        
        # Stream response
        async for chunk in self._stream_response(query, context_str, history_str):
            yield chunk
    
    def _create_llm(self) -> Any:
        """Create appropriate LLM based on configuration."""
        if self.config.provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.streaming
            )
        elif self.config.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.streaming
            )
        elif self.config.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                streaming=self.config.streaming
            )
        elif self.config.provider == "mistral":
            from langchain_mistralai import ChatMistralAI
            return ChatMistralAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.streaming
            )
        elif self.config.provider == "cohere":
            from langchain_cohere import ChatCohere
            return ChatCohere(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
        else:
            # Default to OpenAI
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.streaming
            )
    
    def _generate_chat_response(self, query: str, context: str, history: str = "") -> str:
        """Generate response using chat format."""
        # Create messages
        messages = []
        
        # Add system message
        system_prompt = self.config.system_prompt or self.default_system_prompt
        messages.append(SystemMessage(content=system_prompt))
        
        # Add context message
        context_message = f"""Use the following context to answer the user's question:
        
        {context}
        
        Remember to only use information from the provided context and not to make up information."""
        messages.append(SystemMessage(content=context_message))
        
        # Add chat history if available
        if history:
            messages.append(SystemMessage(content=f"Previous conversation:\n{history}"))
        
        # Add user query
        messages.append(HumanMessage(content=query))
        
        # Get response
        response = self.llm.invoke(messages)
        return response.content
    
    def _generate_completion_response(self, query: str, context: str, history: str = "") -> str:
        """Generate response using completion format."""
        # Create prompt
        system_prompt = self.config.system_prompt or self.default_system_prompt
        
        prompt_text = f"""{system_prompt}

Context:
{context}

"""
        
        if history:
            prompt_text += f"""Previous conversation:
{history}

"""
        
        prompt_text += f"""User query: {query}

Assistant response:"""
        
        # Create prompt template and chain
        prompt = PromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm | StrOutputParser()
        
        # Get response
        return chain.invoke({})
    
    async def _stream_response(self, query: str, context: str, history: str = "") -> Generator[str, None, None]:
        """Stream response generation."""
        # Create messages for chat
        messages = []
        
        # Add system message
        system_prompt = self.config.system_prompt or self.default_system_prompt
        messages.append(SystemMessage(content=system_prompt))
        
        # Add context message
        context_message = f"""Use the following context to answer the user's question:
        
        {context}
        
        Remember to only use information from the provided context and not to make up information."""
        messages.append(SystemMessage(content=context_message))
        
        # Add chat history if available
        if history:
            messages.append(SystemMessage(content=f"Previous conversation:\n{history}"))
        
        # Add user query
        messages.append(HumanMessage(content=query))
        
        # Stream response
        async for chunk in self.llm.astream(messages):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context string from documents."""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Format document content
            content = doc.page_content.strip()
            
            # Format metadata if available
            metadata_str = self._format_metadata(doc.metadata) if doc.metadata else ""
            
            # Add to context parts
            if metadata_str:
                context_parts.append(f"Document {i+1} (Source: {metadata_str}):\n{content}")
            else:
                context_parts.append(f"Document {i+1}:\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format document metadata."""
        # Try to extract source information
        source = metadata.get("source", "")
        
        # If no source but we have a file path, use that
        if not source and "file_path" in metadata:
            source = metadata["file_path"]
        
        # If we have a page number, add it
        if "page" in metadata:
            source = f"{source}, page {metadata['page']}"
        
        return source
    
    def _post_process_response(self, response: str) -> str:
        """Post-process generated response."""
        # Clean up response
        response = response.strip()
        
        # Remove common prefixes like "I'll answer based on the context"
        prefixes_to_remove = [
            "Based on the provided context, ",
            "According to the context, ",
            "From the context provided, ",
            "The context indicates that ",
            "As mentioned in the context, "
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):]
                break
        
        return response.strip()