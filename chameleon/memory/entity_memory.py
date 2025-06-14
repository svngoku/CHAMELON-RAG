from typing import List, Dict, Any, Optional, Set, Tuple
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from datetime import datetime

from ..base import BaseMemory, MemoryConfig


class EntityMemory(BaseMemory):
    """
    Advanced memory system that tracks entities mentioned in conversations.
    
    Features:
    - Extracts and tracks entities from conversations
    - Maintains entity-specific context
    - Allows retrieval of entity-relevant history
    - Provides temporal awareness with timestamps
    
    Based on LangChain's entity memory concepts:
    https://python.langchain.com/docs/modules/memory/types/entity_summary_memory
    """
    
    def __init__(
        self, 
        config: MemoryConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize entity memory.
        
        Args:
            config: Memory configuration
            llm: Language model for entity extraction
        """
        super().__init__(config)
        self.llm = llm or ChatOpenAI(temperature=0)
        self.conversation_memory = ConversationBufferMemory(
            return_messages=config.return_messages,
            memory_key="chat_history"
        )
        self.entity_store = {}  # Maps entity -> information
        self.entity_mentions = {}  # Maps entity -> list of conversation indexes
        self.conversation_history = []
        self.conversation_timestamps = []
    
    def add(self, query: str, response: Dict[str, Any]) -> None:
        """
        Add an interaction to memory and extract entities.
        
        Args:
            query: User query
            response: System response (dict with 'response' key)
        """
        response_text = response.get('response', '')
        
        # Store conversation with timestamp
        timestamp = datetime.now().isoformat()
        self.conversation_history.append({
            "query": query,
            "response": response_text,
            "timestamp": timestamp
        })
        self.conversation_timestamps.append(timestamp)
        
        # Add to standard conversation memory
        self.conversation_memory.save_context(
            {"input": query}, 
            {"output": response_text}
        )
        
        # Extract entities from the conversation
        conversation = f"User: {query}\nAI: {response_text}"
        entities = self._extract_entities(conversation)
        
        # Update entity store and mentions
        for entity, summary in entities:
            conversation_idx = len(self.conversation_history) - 1
            
            # Update entity information
            if entity in self.entity_store:
                self.entity_store[entity] = self._merge_entity_information(
                    self.entity_store[entity], 
                    summary
                )
            else:
                self.entity_store[entity] = summary
            
            # Track entity mentions
            if entity in self.entity_mentions:
                self.entity_mentions[entity].append(conversation_idx)
            else:
                self.entity_mentions[entity] = [conversation_idx]
    
    def get(self, k: Optional[int] = None, entities: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve memory contents, optionally filtered by entities.
        
        Args:
            k: Number of most recent interactions to return (None for all)
            entities: List of entities to filter by (None for all)
            
        Returns:
            List of conversation entries
        """
        # Get recent k interactions if specified
        history = self.conversation_history
        if k is not None:
            history = history[-k:]
        
        # If no entity filter, return regular history
        if not entities:
            return history
        
        # Filter history by mentioned entities
        relevant_indices = set()
        for entity in entities:
            if entity in self.entity_mentions:
                relevant_indices.update(self.entity_mentions[entity])
        
        # Get conversations that mention the entities
        entity_history = [
            self.conversation_history[i] 
            for i in sorted(relevant_indices) 
            if i < len(self.conversation_history)
        ]
        
        # Limit to k if specified
        if k is not None:
            entity_history = entity_history[-k:]
        
        return entity_history
    
    def get_entity_info(self, entity: str) -> Optional[str]:
        """Get stored information about a specific entity."""
        return self.entity_store.get(entity)
    
    def get_relevant_entities(self, query: str, threshold: int = 3) -> List[str]:
        """
        Find entities from memory that are relevant to the current query.
        
        Args:
            query: Current user query
            threshold: Minimum relevance score to include an entity
            
        Returns:
            List of relevant entity names
        """
        query_entities = self._extract_entities_from_text(query)
        relevant_entities = list(query_entities)
        
        # Find additional entities that might be contextually relevant
        # based on co-occurrence patterns
        for query_entity in query_entities:
            if query_entity in self.entity_mentions:
                # Find conversations where this entity was mentioned
                conversations = self.entity_mentions[query_entity]
                
                # Find other entities mentioned in these conversations
                for entity, mentions in self.entity_mentions.items():
                    if entity != query_entity and entity not in relevant_entities:
                        # Count co-occurrences
                        co_occurrences = len(set(conversations) & set(mentions))
                        if co_occurrences >= threshold:
                            relevant_entities.append(entity)
        
        return relevant_entities
    
    def clear(self) -> None:
        """Clear all memory."""
        self.conversation_memory.clear()
        self.entity_store.clear()
        self.entity_mentions.clear()
        self.conversation_history.clear()
        self.conversation_timestamps.clear()
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities and their information from text.
        
        Returns:
            List of (entity, information) tuples
        """
        prompt = f"""Extract the key entities from the following conversation along with a brief description of each entity based ONLY on the information in the conversation. 
        For each entity, provide:
        1. The entity name
        2. A concise description using ONLY information explicitly stated in the conversation
        
        Format: Entity: Description
        
        Conversation:
        {text}
        
        Entities:"""
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            
            # Parse the response to extract entity-description pairs
            entities = []
            for line in response.split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    entity = parts[0].strip()
                    description = parts[1].strip()
                    if entity and description:
                        entities.append((entity, description))
            
            return entities
        except Exception:
            return []
    
    def _extract_entities_from_text(self, text: str) -> Set[str]:
        """Extract just entity names from text."""
        prompt = f"""Extract the key entities (people, places, organizations, concepts) from the following text. 
        Output ONLY a comma-separated list of entity names, nothing else.
        
        Text: {text}
        
        Entities:"""
        
        try:
            response = self.llm.invoke(prompt).content.strip()
            entities = {
                entity.strip() 
                for entity in response.split(',') 
                if entity.strip()
            }
            return entities
        except Exception:
            return set()
    
    def _merge_entity_information(self, existing: str, new_info: str) -> str:
        """
        Merge existing entity information with new information.
        
        Uses the LLM to create a coherent, updated description.
        """
        prompt = f"""Merge these two descriptions of the same entity into a single, coherent description.
        Include all non-redundant information.
        Keep the result concise.
        
        Description 1: {existing}
        Description 2: {new_info}
        
        Merged description:"""
        
        try:
            merged = self.llm.invoke(prompt).content.strip()
            return merged
        except Exception:
            # If merging fails, concatenate with a separator
            return f"{existing}; {new_info}"