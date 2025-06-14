# CHAMELEON-RAG

![CHAMELEON-RAG](/docs/assets/CHAMELEON.webp)

CHAMELEON-RAG is a flexible framework for implementing advanced Retrieval-Augmented Generation (RAG) techniques, inspired by the latest features and best practices from LangChain.

## 🎯 Key Features

- **Multiple RAG Types**: Basic, Contextual, Multi-Query, and Parent Document RAG
- **Flexible LLM Support**: OpenAI, Together AI, Groq, Mistral, Cohere, Google Vertex AI, **LiteLLM**, and **OpenRouter**
- **Vector Store Options**: FAISS, Chroma, Weaviate, and Pinecone
- **Advanced Components**: Query transformation, contextual compression, entity memory
- **Production Ready**: Health monitoring, structure validation, comprehensive testing
- **Easy Integration**: Builder pattern with predefined templates

## Features

- **Advanced Retrieval Techniques**
  - Multi-query retrieval for improved recall
  - Parent document retrieval for complete context
  - Hybrid retrieval combining semantic and keyword search
  - Dynamic reranking of retrieved documents

- **Intelligent Preprocessing**
  - Query transformation and expansion
  - Semantic chunking for context-aware document splits
  - Multiple chunking strategies (fixed size, semantic, etc.)

- **Enhanced Context Processing**
  - Contextual compression to filter irrelevant content
  - Entity-aware memory for conversation tracking
  - Temporal awareness in conversation history

- **Adaptive Generation**
  - Support for multiple LLM providers
  - Streaming support for real-time responses
  - Different chain types (stuff, map-reduce, refine, map-rerank)

- **Evaluation and Monitoring**
  - Built-in evaluation metrics (relevance, faithfulness, etc.)
  - Performance monitoring and tracing
  - Hallucination detection

- **Tool Integration**
  - External tool calling based on query requirements
  - Web search integration for supplementing static knowledge

## Installation

```bash
pip install chameleon-rag
```

Or install from source:

```bash
git clone https://github.com/yourusername/CHAMELEON-RAG.git
cd CHAMELEON-RAG
pip install -e .
```

## Quick Start

```python
from chameleon.pipeline.enhanced_rag_pipeline import EnhancedRAGPipeline
from chameleon.base import PipelineConfig, RetrieverConfig, GeneratorConfig
from langchain_core.documents import Document

# Create your documents
documents = [
    Document(page_content="RAG combines retrieval with generation for more accurate responses.",
             metadata={"source": "intro_to_rag.txt"}),
    Document(page_content="Large language models can hallucinate when lacking relevant context.",
             metadata={"source": "llm_limitations.txt"})
]

# Configure your pipeline
config = PipelineConfig(
    retriever_config=RetrieverConfig(
        top_k=3,
        retrieval_type="hybrid",
        reranking_enabled=True,
        multi_query_enabled=True
    ),
    generator_config=GeneratorConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.7
    ),
    chain_type="stuff"
)

# Create the pipeline
pipeline = EnhancedRAGPipeline(
    title="My RAG Pipeline",
    documents=documents,
    config=config,
    enable_evaluation=True
)

# Run a query
response = pipeline.run("What is RAG and why is it useful?")
print(response["response"])
```

## Examples

See the `examples/` directory for more detailed examples:

- `examples/rag_example.py`: Basic RAG pipeline usage
- `examples/enhanced_rag_example.py`: Advanced RAG features

## Architecture

CHAMELEON-RAG follows a modular design with the following key components:

1. **Retrievers**: Responsible for finding relevant documents
   - AdvancedRetriever: Hybrid semantic and keyword search
   - MultiQueryRetriever: Improves recall with query variations
   - ParentDocumentRetriever: Maintains parent-child relationships

2. **Preprocessors**: Prepare queries and documents
   - QueryTransformer: Expands and transforms queries
   - SemanticChunker: Content-aware document splitting

3. **Postprocessors**: Refine retrieved documents
   - ContextualCompressor: Filters irrelevant content
   - Reranker: Reorders documents by relevance

4. **Memory**: Manages conversation context
   - EntityMemory: Tracks entities across conversations
   - MemoryAdapter: Standard conversation buffer

5. **Generators**: Produce responses from context
   - AdvancedGenerator: Multi-provider LLM support
   - StreamingGenerator: Real-time response streaming

6. **Pipeline**: Orchestrates the entire RAG process
   - EnhancedRAGPipeline: Combines all components with advanced features
   - Different chain types for various use cases

## Customization

CHAMELEON-RAG is designed to be easily customizable:

- Create custom retrievers by extending `BaseRetriever`
- Implement custom generators by extending `BaseGenerator`
- Add new preprocessing techniques by extending `BasePreprocessor`
- Create custom memory systems by extending `BaseMemory`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### LiteLLM Support (100+ Providers)

LiteLLM provides access to 100+ LLM providers through a unified interface:

```python
from chameleon.utils.pipeline_builder import PipelineBuilder

# Anthropic Claude
pipeline = (PipelineBuilder()
            .with_litellm("anthropic/claude-3-sonnet-20240229")
            .with_chroma()
            .build())

# Google Gemini
pipeline = (PipelineBuilder()
            .with_litellm("gemini/gemini-pro")
            .with_faiss()
            .build())

# Cohere Command
pipeline = (PipelineBuilder()
            .with_litellm("cohere/command-r-plus")
            .with_chroma()
            .build())
```

### OpenRouter Support

OpenRouter provides access to multiple AI models through a single API:

```python
# Claude 3.5 Sonnet via OpenRouter
pipeline = (PipelineBuilder()
            .with_openrouter("anthropic/claude-3.5-sonnet")
            .with_chroma()
            .build())

# Llama 3.1 405B via OpenRouter
pipeline = (PipelineBuilder()
            .with_openrouter("meta-llama/llama-3.1-405b-instruct")
            .with_faiss()
            .build())

# GPT-4 Turbo via OpenRouter
pipeline = (PipelineBuilder()
            .with_openrouter("openai/gpt-4-turbo")
            .with_chroma()
            .build())
```

### Environment Variables

Set the appropriate API keys:

```bash
# Core providers
export OPENAI_API_KEY="your-openai-key"
export TOGETHER_API_KEY="your-together-key"
export GROQ_API_KEY="your-groq-key"

# Extended providers
export OPENROUTER_API_KEY="your-openrouter-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  # For LiteLLM
export GOOGLE_API_KEY="your-google-key"        # For LiteLLM
```