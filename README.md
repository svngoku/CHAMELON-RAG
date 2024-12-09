# CHAMELEON RAG: CHAinable Multi-technique Extensible Library for Enhanced ON-demand Retrieval-Augmented Generation

![CHAMELEON RAG Logo](/docs/assets/CHAMELEON.webp)

## Abstract

`CHAMELEON RAG` is a versatile library designed to enhance retrieval-augmented generation (RAG) processes. It provides a flexible framework for integrating multiple retrieval and generation techniques, allowing users to build custom pipelines for various applications.

## Features

- **Multiple RAG Techniques**: Support for Naive, Modular, and Advanced RAG implementations
- **Configurable Components**: Easily swap and configure retrievers, generators, and preprocessors
- **Memory Management**: Built-in support for different types of conversation memory
- **Advanced Features**: Re-ranking, filtering, and enhanced generation capabilities
- **Easy Integration**: Simple API for building custom RAG pipelines

## Installation

To install CHAMELEON RAG, clone the repository and install the required dependencies:

```bash
git clone https://github.com/svngoku/chamelon-rag.git
cd chamelon-rag
pip install -r requirements.txt
```

## Usage

Here's how to use CHAMELEON RAG with different techniques:

```python
from setup import PipelineFactory
from rag_techniques.loaders import FileLoader

# Load your documents
loader = FileLoader()
documents = loader.load_text_file('path/to/your/file.txt')

# 1. Naive RAG (Simple and straightforward)
naive_pipeline = PipelineFactory.create_pipeline(
    documents=documents,
    rag_type="naive"
)

# 2. Modular RAG (Configurable components)
modular_pipeline = PipelineFactory.create_pipeline(
    documents=documents,
    rag_type="modular",
    retriever_config={"top_k": 3},
    generator_config={"temperature": 0.7}
)

# 3. Advanced RAG (Enhanced capabilities)
advanced_pipeline = PipelineFactory.create_pipeline(
    documents=documents,
    rag_type="advanced",
    retriever_config={
        "top_k": 5,
        "reranking_enabled": True,
        "filtering_threshold": 0.8
    },
    generator_config={
        "temperature": 0.5,
        "max_tokens": 500
    }
)

# Run queries
results = pipeline.run(query="Your query here", documents=documents)
print(results['response'])
```

## RAG Techniques

1. **Naive RAG**: Basic implementation for simple use cases
   - Direct retrieval and generation
   - No advanced preprocessing
   - Suitable for straightforward Q&A

2. **Modular RAG**: Flexible implementation with swappable components
   - Configurable retrievers and generators
   - Custom preprocessing pipeline
   - Memory management options

3. **Advanced RAG**: Enhanced implementation with advanced features
   - Re-ranking of retrieved documents
   - Advanced filtering and preprocessing
   - Optimized generation parameters
   - Context-aware responses

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
