"""
Pipeline Builder Utility for CHAMELEON RAG Framework
Provides easy-to-use builders for creating optimized RAG pipelines.
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

from ..base import (
    PipelineConfig, RetrieverConfig, GeneratorConfig, MemoryConfig,
    BaseRetriever, BaseGenerator, BaseMemory, BasePreprocessor, BasePostprocessor
)
from ..pipeline.rag_pipeline import RAGPipeline
from ..pipeline.enhanced_rag_pipeline import EnhancedRAGPipeline
from ..utils.logging_utils import setup_colored_logger, COLORS
from langchain_core.documents import Document

class RAGType(Enum):
    """Supported RAG pipeline types."""
    BASIC = "basic"
    MODULAR = "modular" 
    ENHANCED = "enhanced"
    ADVANCED = "advanced"

class Provider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    TOGETHER = "together"
    GROQ = "groq"
    MISTRAL = "mistral"
    COHERE = "cohere"
    VERTEXAI = "vertexai"
    LITELLM = "litellm"
    OPENROUTER = "openrouter"

class VectorStore(Enum):
    """Supported vector stores."""
    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"

@dataclass
class PipelineTemplate:
    """Template for common pipeline configurations."""
    name: str
    description: str
    rag_type: RAGType
    retriever_config: Dict[str, Any]
    generator_config: Dict[str, Any]
    memory_config: Dict[str, Any]
    use_cases: List[str] = field(default_factory=list)

class ChameleonPipelineBuilder:
    """Builder for creating optimized CHAMELEON RAG pipelines."""
    
    def __init__(self):
        self.logger = setup_colored_logger()
        self.reset()
        
        # Predefined templates
        self.templates = {
            "quick_start": PipelineTemplate(
                name="Quick Start",
                description="Simple RAG pipeline for getting started",
                rag_type=RAGType.MODULAR,
                retriever_config={
                    "top_k": 3,
                    "similarity_threshold": 0.7,
                    "retrieval_type": "semantic"
                },
                generator_config={
                    "model_name": "gpt-3.5-turbo",
                    "provider": "openai",
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                memory_config={
                    "memory_type": "buffer",
                    "max_history": 5
                },
                use_cases=["Q&A", "Simple chat", "Document search"]
            ),
            
            "production_ready": PipelineTemplate(
                name="Production Ready",
                description="Optimized pipeline for production use",
                rag_type=RAGType.ENHANCED,
                retriever_config={
                    "top_k": 5,
                    "similarity_threshold": 0.3,
                    "retrieval_type": "semantic",
                    "reranking_enabled": True,
                    "filtering_enabled": True,
                    "filtering_threshold": 0.2,
                    "multi_query_enabled": True
                },
                generator_config={
                    "model_name": "gpt-4o-mini",
                    "provider": "openai",
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                memory_config={
                    "memory_type": "entity",
                    "max_history": 15
                },
                use_cases=["Production apps", "Complex Q&A", "Research assistance"]
            ),
            
            "cost_optimized": PipelineTemplate(
                name="Cost Optimized",
                description="Budget-friendly pipeline with good performance",
                rag_type=RAGType.MODULAR,
                retriever_config={
                    "top_k": 3,
                    "similarity_threshold": 0.5,
                    "retrieval_type": "semantic",
                    "embedding_model": "text-embedding-3-small"
                },
                generator_config={
                    "model_name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    "provider": "together",
                    "temperature": 0.5,
                    "max_tokens": 1500
                },
                memory_config={
                    "memory_type": "buffer",
                    "max_history": 8
                },
                use_cases=["Budget projects", "Prototyping", "Educational use"]
            ),
            
            "research_focused": PipelineTemplate(
                name="Research Focused",
                description="High-accuracy pipeline for research and analysis",
                rag_type=RAGType.ENHANCED,
                retriever_config={
                    "top_k": 8,
                    "similarity_threshold": 0.2,
                    "retrieval_type": "semantic",
                    "reranking_enabled": True,
                    "filtering_enabled": True,
                    "filtering_threshold": 0.1,
                    "multi_query_enabled": True,
                    "parent_document_enabled": True
                },
                generator_config={
                    "model_name": "gpt-4o",
                    "provider": "openai",
                    "temperature": 0.1,
                    "max_tokens": 3000
                },
                memory_config={
                    "memory_type": "entity",
                    "max_history": 20
                },
                use_cases=["Academic research", "Legal analysis", "Medical Q&A"]
            )
        }
    
    def reset(self):
        """Reset builder to default state."""
        self._title = "CHAMELEON RAG Pipeline"
        self._rag_type = RAGType.MODULAR
        self._documents = []
        self._retriever_config = {}
        self._generator_config = {}
        self._memory_config = {}
        self._preprocessors = []
        self._postprocessors = []
        self._tools = []
        self._custom_components = {}
        return self
    
    def from_template(self, template_name: str):
        """Initialize builder from a predefined template."""
        if template_name not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
        
        template = self.templates[template_name]
        self.logger.info(f"{COLORS['BLUE']}📋 Using template: {template.name}{COLORS['ENDC']}")
        self.logger.info(f"Description: {template.description}")
        self.logger.info(f"Use cases: {', '.join(template.use_cases)}")
        
        self._rag_type = template.rag_type
        self._retriever_config = template.retriever_config.copy()
        self._generator_config = template.generator_config.copy()
        self._memory_config = template.memory_config.copy()
        
        return self
    
    def with_title(self, title: str):
        """Set pipeline title."""
        self._title = title
        return self
    
    def with_documents(self, documents: List[Document]):
        """Set documents for the pipeline."""
        self._documents = documents
        self.logger.info(f"{COLORS['GREEN']}📚 Added {len(documents)} documents{COLORS['ENDC']}")
        return self
    
    def with_rag_type(self, rag_type: Union[RAGType, str]):
        """Set RAG pipeline type."""
        if isinstance(rag_type, str):
            rag_type = RAGType(rag_type)
        self._rag_type = rag_type
        return self
    
    def with_retriever(self, **config):
        """Configure retriever settings."""
        self._retriever_config.update(config)
        return self
    
    def with_generator(self, **config):
        """Configure generator settings."""
        self._generator_config.update(config)
        return self
    
    def with_memory(self, **config):
        """Configure memory settings."""
        self._memory_config.update(config)
        return self
    
    def with_vector_store(self, store_type: Union[VectorStore, str], **config):
        """Configure vector store."""
        if isinstance(store_type, str):
            store_type = VectorStore(store_type)
        
        self._retriever_config.update({
            "store_type": store_type.value,
            **config
        })
        return self
    
    def with_llm(self, 
                 model_name: str, 
                 provider: Union[Provider, str],
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 **kwargs):
        """Configure LLM settings."""
        if isinstance(provider, str):
            provider = Provider(provider)
        
        self._generator_config.update({
            "model_name": model_name,
            "provider": provider.value,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        })
        return self
    
    def with_openai(self, model_name: str = "gpt-4o-mini", **kwargs):
        """Quick configuration for OpenAI models."""
        return self.with_llm(model_name, Provider.OPENAI, **kwargs)
    
    def with_together(self, model_name: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", **kwargs):
        """Quick configuration for Together AI models."""
        return self.with_llm(model_name, Provider.TOGETHER, **kwargs)
    
    def with_litellm(self, model_name: str, **kwargs):
        """Quick configuration for LiteLLM (supports multiple providers).
        
        Examples:
        - Anthropic: "anthropic/claude-3-sonnet-20240229"
        - OpenAI: "openai/gpt-4"
        - Cohere: "cohere/command-r-plus"
        - Gemini: "gemini/gemini-pro"
        """
        return self.with_llm(model_name, Provider.LITELLM, **kwargs)
    
    def with_openrouter(self, model_name: str = "anthropic/claude-3.5-sonnet", **kwargs):
        """Quick configuration for OpenRouter models.
        
        Popular OpenRouter models:
        - "anthropic/claude-3.5-sonnet"
        - "openai/gpt-4-turbo"
        - "meta-llama/llama-3.1-405b-instruct"
        - "google/gemini-pro-1.5"
        """
        return self.with_llm(model_name, Provider.OPENROUTER, **kwargs)
    
    def with_groq(self, model_name: str = "llama3-8b-8192", **kwargs):
        """Quick configuration for Groq models."""
        return self.with_llm(model_name, Provider.GROQ, **kwargs)
    
    def with_faiss(self, **kwargs):
        """Quick configuration for FAISS vector store."""
        return self.with_vector_store(VectorStore.FAISS, **kwargs)
    
    def with_chroma(self, **kwargs):
        """Quick configuration for Chroma vector store."""
        return self.with_vector_store(VectorStore.CHROMA, **kwargs)
    
    def with_pinecone(self, **kwargs):
        """Quick configuration for Pinecone vector store."""
        return self.with_vector_store(VectorStore.PINECONE, **kwargs)
    
    def with_weaviate(self, **kwargs):
        """Quick configuration for Weaviate vector store."""
        return self.with_vector_store(VectorStore.WEAVIATE, **kwargs)
    
    def with_basic_rag(self):
        """Configure for basic RAG."""
        return self.with_rag_type(RAGType.BASIC)
    
    def with_modular_rag(self):
        """Configure for modular RAG."""
        return self.with_rag_type(RAGType.MODULAR)
    
    def with_enhanced_rag(self):
        """Configure for enhanced RAG."""
        return self.with_rag_type(RAGType.ENHANCED)
    
    def with_contextual_rag(self):
        """Configure for contextual RAG (enhanced)."""
        return self.with_rag_type(RAGType.ENHANCED)
    
    def with_multi_query_rag(self):
        """Configure for multi-query RAG (enhanced)."""
        return self.with_rag_type(RAGType.ENHANCED).with_retriever(multi_query_enabled=True)
    
    def with_parent_document_rag(self):
        """Configure for parent document RAG (enhanced)."""
        return self.with_rag_type(RAGType.ENHANCED).with_retriever(parent_document_enabled=True)
    
    def with_preprocessing(self, *preprocessors: BasePreprocessor):
        """Add preprocessing components."""
        self._preprocessors.extend(preprocessors)
        return self
    
    def with_postprocessing(self, *postprocessors: BasePostprocessor):
        """Add postprocessing components."""
        self._postprocessors.extend(postprocessors)
        return self
    
    def with_tools(self, *tools):
        """Add tools to the pipeline."""
        self._tools.extend(tools)
        return self
    
    def optimize_for_use_case(self, use_case: str):
        """Optimize pipeline configuration for specific use cases."""
        optimizations = {
            "qa": {
                "retriever": {"top_k": 3, "similarity_threshold": 0.7},
                "generator": {"temperature": 0.3, "max_tokens": 1000}
            },
            "chat": {
                "retriever": {"top_k": 4, "similarity_threshold": 0.6},
                "generator": {"temperature": 0.7, "max_tokens": 1500},
                "memory": {"memory_type": "buffer", "max_history": 10}
            },
            "research": {
                "retriever": {"top_k": 8, "similarity_threshold": 0.2, "reranking_enabled": True},
                "generator": {"temperature": 0.1, "max_tokens": 3000},
                "memory": {"memory_type": "entity", "max_history": 20}
            },
            "summarization": {
                "retriever": {"top_k": 10, "similarity_threshold": 0.3},
                "generator": {"temperature": 0.3, "max_tokens": 2000}
            },
            "creative": {
                "retriever": {"top_k": 5, "similarity_threshold": 0.5},
                "generator": {"temperature": 0.9, "max_tokens": 2000}
            }
        }
        
        if use_case.lower() in optimizations:
            opt = optimizations[use_case.lower()]
            if "retriever" in opt:
                self._retriever_config.update(opt["retriever"])
            if "generator" in opt:
                self._generator_config.update(opt["generator"])
            if "memory" in opt:
                self._memory_config.update(opt["memory"])
            
            self.logger.info(f"{COLORS['BLUE']}🎯 Optimized for {use_case} use case{COLORS['ENDC']}")
        else:
            available = list(optimizations.keys())
            self.logger.warning(f"{COLORS['YELLOW']}⚠️ Unknown use case '{use_case}'. Available: {available}{COLORS['ENDC']}")
        
        return self
    
    def validate_configuration(self) -> List[str]:
        """Validate the current configuration and return any issues."""
        issues = []
        
        # Check required components
        if not self._documents:
            issues.append("No documents provided")
        
        if not self._generator_config.get("model_name"):
            issues.append("No model specified for generator")
        
        if not self._generator_config.get("provider"):
            issues.append("No provider specified for generator")
        
        # Check configuration consistency
        if self._rag_type == RAGType.ENHANCED:
            if not self._retriever_config.get("reranking_enabled"):
                issues.append("Enhanced RAG should typically use reranking")
        
        # Check memory configuration
        memory_type = self._memory_config.get("memory_type")
        if memory_type == "entity" and self._memory_config.get("max_history", 0) < 10:
            issues.append("Entity memory works better with higher max_history (>=10)")
        
        return issues
    
    def build(self, validate: bool = True) -> Union[RAGPipeline, EnhancedRAGPipeline]:
        """Build the configured RAG pipeline."""
        if validate:
            issues = self.validate_configuration()
            if issues:
                self.logger.warning(f"{COLORS['YELLOW']}⚠️ Configuration issues found:{COLORS['ENDC']}")
                for issue in issues:
                    self.logger.warning(f"  • {issue}")
                
                response = input("Continue building pipeline? (y/n): ")
                if response.lower() != 'y':
                    raise ValueError("Pipeline building cancelled due to configuration issues")
        
        self.logger.info(f"{COLORS['BLUE']}🔧 Building {self._rag_type.value} RAG pipeline...{COLORS['ENDC']}")
        
        # Create configurations
        retriever_config = RetrieverConfig(**self._retriever_config)
        generator_config = GeneratorConfig(**self._generator_config)
        memory_config = MemoryConfig(**self._memory_config)
        
        pipeline_config = PipelineConfig(
            rag_type=self._rag_type.value,
            retriever_config=retriever_config,
            generator_config=generator_config,
            memory_config=memory_config
        )
        
        # Build appropriate pipeline type
        if self._rag_type in [RAGType.ENHANCED, RAGType.ADVANCED]:
            pipeline = EnhancedRAGPipeline(
                title=self._title,
                documents=self._documents,
                config=pipeline_config,
                preprocessors=self._preprocessors,
                postprocessors=self._postprocessors,
                tools=self._tools,
                enable_evaluation=False
            )
        else:
            pipeline = RAGPipeline(
                title=self._title,
                documents=self._documents,
                config=pipeline_config,
                preprocessors=self._preprocessors
            )
        
        self.logger.info(f"{COLORS['GREEN']}✅ Pipeline built successfully!{COLORS['ENDC']}")
        return pipeline
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            "title": self._title,
            "rag_type": self._rag_type.value,
            "documents_count": len(self._documents),
            "retriever_config": self._retriever_config,
            "generator_config": self._generator_config,
            "memory_config": self._memory_config,
            "preprocessors_count": len(self._preprocessors),
            "postprocessors_count": len(self._postprocessors),
            "tools_count": len(self._tools)
        }
    
    def print_configuration(self):
        """Print the current configuration in a readable format."""
        config = self.get_configuration_summary()
        
        print(f"\n{COLORS['BLUE']}📋 PIPELINE CONFIGURATION{COLORS['ENDC']}")
        print("=" * 50)
        print(f"Title: {config['title']}")
        print(f"RAG Type: {config['rag_type']}")
        print(f"Documents: {config['documents_count']}")
        
        print(f"\n{COLORS['GREEN']}🔍 Retriever:{COLORS['ENDC']}")
        for key, value in config['retriever_config'].items():
            print(f"  • {key}: {value}")
        
        print(f"\n{COLORS['GREEN']}🤖 Generator:{COLORS['ENDC']}")
        for key, value in config['generator_config'].items():
            print(f"  • {key}: {value}")
        
        print(f"\n{COLORS['GREEN']}🧠 Memory:{COLORS['ENDC']}")
        for key, value in config['memory_config'].items():
            print(f"  • {key}: {value}")
        
        if config['preprocessors_count'] > 0:
            print(f"\n{COLORS['CYAN']}⚙️ Preprocessors: {config['preprocessors_count']}{COLORS['ENDC']}")
        
        if config['postprocessors_count'] > 0:
            print(f"{COLORS['CYAN']}⚙️ Postprocessors: {config['postprocessors_count']}{COLORS['ENDC']}")
        
        if config['tools_count'] > 0:
            print(f"{COLORS['CYAN']}🛠️ Tools: {config['tools_count']}{COLORS['ENDC']}")
    
    @classmethod
    def list_templates(cls):
        """List all available templates."""
        builder = cls()
        print(f"\n{COLORS['BLUE']}📋 AVAILABLE TEMPLATES{COLORS['ENDC']}")
        print("=" * 50)
        
        for name, template in builder.templates.items():
            print(f"\n{COLORS['GREEN']}{template.name}{COLORS['ENDC']} ({name})")
            print(f"  Description: {template.description}")
            print(f"  RAG Type: {template.rag_type.value}")
            print(f"  Use Cases: {', '.join(template.use_cases)}")

# Convenience functions
def quick_pipeline(documents: List[Document], 
                  model_name: str = "gpt-4o-mini",
                  provider: str = "openai") -> RAGPipeline:
    """Create a quick RAG pipeline with minimal configuration."""
    return (ChameleonPipelineBuilder()
            .from_template("quick_start")
            .with_documents(documents)
            .with_llm(model_name, provider)
            .build())

def production_pipeline(documents: List[Document],
                       model_name: str = "gpt-4o-mini",
                       provider: str = "openai") -> EnhancedRAGPipeline:
    """Create a production-ready RAG pipeline."""
    return (ChameleonPipelineBuilder()
            .from_template("production_ready")
            .with_documents(documents)
            .with_llm(model_name, provider)
            .build())

def research_pipeline(documents: List[Document],
                     model_name: str = "gpt-4o",
                     provider: str = "openai") -> EnhancedRAGPipeline:
    """Create a research-focused RAG pipeline."""
    return (ChameleonPipelineBuilder()
            .from_template("research_focused")
            .with_documents(documents)
            .with_llm(model_name, provider)
            .build())

if __name__ == "__main__":
    # Demo the builder
    ChameleonPipelineBuilder.list_templates() 