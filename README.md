# CHAMELEON RAG: CHAinable Multi-technique Extensible Library for Enhanced ON-demand Retrieval-Augmented Generation


![CHAMELEON RAG Logo](/docs/assets/CHAMELEON.webp)

## Abstract

`CHAMELEON RAG` is a versatile library designed to enhance retrieval-augmented generation (RAG) processes. It provides a flexible framework for integrating multiple retrieval and generation techniques, allowing users to build custom pipelines for various applications.

## Installation

To install CHAMELEON RAG, clone the repository and install the required dependencies:

```bash
git clone https://github.com/svngoku/chamelon-rag.git
cd chamelon-rag
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use CHAMELON RAG:

```python
from rag_techniques.pipeline.rag_pipeline import RAGPipeline
from rag_techniques.loaders import FileLoader

# Initialize the pipeline
pipeline = RAGPipeline()

# Load data
loader = FileLoader()
documents = loader.load_text_file('path/to/your/file.txt')

# Run the pipeline
results = pipeline.run(query="Your query here", data=documents)
print(results)
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
