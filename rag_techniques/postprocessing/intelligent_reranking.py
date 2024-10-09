from rag_techniques.base import BasePreprocessor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI


class IntelligentReranking(BasePreprocessor):
    def process(self, data):
        pass

