from typing import List, Dict, Any, Optional
from ..base import BaseMemory, MemoryConfig
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory

class MemoryAdapter(BaseMemory):
    """Adapter for LangChain memory components."""
    
    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.memory = self._create_memory()
    
    def validate_config(self, config: MemoryConfig) -> bool:
        """Validate memory configuration."""
        if config.max_history < 0:
            return False
        if config.memory_type not in ["buffer", "buffer_window", "summary"]:
            return False
        return True
    
    def _create_memory(self) -> BaseChatMessageHistory:
        """Create appropriate memory component based on configuration."""
        if self.config.memory_type == "buffer":
            return ConversationBufferMemory(
                return_messages=self.config.return_messages,
                memory_key="chat_history",
                output_key="output"
            )
        # Add support for other memory types here
        raise ValueError(f"Unsupported memory type: {self.config.memory_type}")
    
    def add(self, query: str, response: Dict[str, Any]) -> None:
        """Add interaction to memory."""
        self.memory.save_context(
            {"input": query},
            {"output": response.get('response', '')}
        )
    
    def get(self, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history."""
        variables = self.memory.load_memory_variables({})
        history = variables.get("chat_history", [])
        
        # Convert to our format
        formatted_history = []
        for message in history:
            formatted_history.append({
                "role": message.type,
                "content": message.content
            })
        
        # Limit history if k is specified
        if k is not None:
            formatted_history = formatted_history[-k:]
        
        return formatted_history
    
    def clear(self) -> None:
        """Clear memory."""
        self.memory.clear() 