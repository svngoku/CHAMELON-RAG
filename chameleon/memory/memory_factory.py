from typing import Optional, Dict, Any
from ..base import BaseMemory, MemoryConfig
from .memory_adapter import MemoryAdapter

class MemoryFactory:
    """Factory for creating memory components."""
    
    def __init__(self, memory_type: str = "buffer", memory_config: Optional[Dict[str, Any]] = None):
        """Initialize memory factory with configuration."""
        self.memory_type = memory_type
        self.memory_config = memory_config or {}
    
    def create_memory(self) -> BaseMemory:
        """Create and configure memory component."""
        config = MemoryConfig(
            memory_type=self.memory_type,
            max_history=self.memory_config.get('max_history', 10),
            return_messages=self.memory_config.get('return_messages', True),
            redis_url=self.memory_config.get('redis_url'),
            storage_type=self.memory_config.get('storage_type', 'in_memory')
        )
        
        return MemoryAdapter(config) 