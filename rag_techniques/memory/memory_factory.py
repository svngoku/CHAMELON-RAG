from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
import logging
from rag_techniques.utils.logging_utils import COLORS

class MemoryFactory(BaseModel):
    """Factory for creating different types of memory components."""
    
    memory_type: str = Field(
        default="buffer",
        description="Type of memory to create (buffer, window, redis)"
    )
    memory_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for memory initialization"
    )
    
    class Config:
        arbitrary_types_allowed = True

    def create_memory(self) -> Any:
        """Create and return a memory instance based on configuration."""
        try:
            memory_types = {
                "buffer": self._create_buffer_memory,
                "window": self._create_window_memory,
                "redis": self._create_redis_memory
            }
            
            if self.memory_type not in memory_types:
                raise ValueError(f"Unsupported memory type: {self.memory_type}")
            
            logging.info(f"{COLORS['BLUE']}Creating {self.memory_type} memory{COLORS['ENDC']}")
            memory = memory_types[self.memory_type]()
            logging.info(f"{COLORS['GREEN']}Successfully created memory{COLORS['ENDC']}")
            
            return memory
            
        except Exception as e:
            logging.error(f"{COLORS['RED']}Error creating memory: {str(e)}{COLORS['ENDC']}")
            raise

    def _create_buffer_memory(self) -> ConversationBufferMemory:
        """Create a basic buffer memory."""
        logging.info(f"{COLORS['BLUE']}Configuring buffer memory{COLORS['ENDC']}")
        return ConversationBufferMemory(
            return_messages=self.memory_config.get("return_messages", True),
            output_key=self.memory_config.get("output_key", "output"),
            memory_key=self.memory_config.get("memory_key", "chat_history")
        )

    def _create_window_memory(self) -> ConversationBufferWindowMemory:
        """Create a windowed memory with configurable size."""
        logging.info(f"{COLORS['BLUE']}Configuring window memory with size {self.memory_config.get('window_size', 5)}{COLORS['ENDC']}")
        return ConversationBufferWindowMemory(
            k=self.memory_config.get("window_size", 5),
            return_messages=self.memory_config.get("return_messages", True),
            output_key=self.memory_config.get("output_key", "output"),
            memory_key=self.memory_config.get("memory_key", "chat_history")
        )

    def _create_redis_memory(self) -> ConversationBufferMemory:
        """Create a Redis-backed memory."""
        session_id = self.memory_config.get("session_id", "default")
        redis_url = self.memory_config.get("redis_url", "redis://localhost:6379")
        
        logging.info(f"{COLORS['BLUE']}Configuring Redis memory with session {session_id}{COLORS['ENDC']}")
        
        message_history = RedisChatMessageHistory(
            session_id=session_id,
            url=redis_url
        )
        
        return ConversationBufferMemory(
            chat_memory=message_history,
            return_messages=self.memory_config.get("return_messages", True),
            output_key=self.memory_config.get("output_key", "output"),
            memory_key=self.memory_config.get("memory_key", "chat_history")
        )

    @classmethod
    def create_default_memory(cls) -> ConversationBufferMemory:
        """Create a memory instance with default settings."""
        return cls(memory_type="buffer").create_memory()

    @classmethod
    def create_redis_memory_with_session(cls, session_id: str, redis_url: Optional[str] = None) -> ConversationBufferMemory:
        """Create a Redis memory instance with specific session ID."""
        config = {
            "session_id": session_id,
            "return_messages": True
        }
        if redis_url:
            config["redis_url"] = redis_url
            
        return cls(
            memory_type="redis",
            memory_config=config
        ).create_memory() 