from typing import List, Any, Dict, Optional
from chameleon.base import BaseRetriever, RetrieverConfig
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from dataclasses import asdict

class AdvancedRetriever(BaseRetriever):
    """Advanced retriever with multiple retrieval methods and re-ranking capabilities."""
    
    def __init__(self, config: RetrieverConfig):
        super().__init__(config)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') if config.reranking_enabled else None
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.document_store = []
        self.document_embeddings = None
        self.bm25 = None
        
    def validate_config(self, config: RetrieverConfig) -> bool:
        """Validate retriever configuration."""
        if config.top_k < 1:
            return False
        if not (0 <= config.similarity_threshold <= 1):
            return False
        if not (0 <= config.filtering_threshold <= 1):
            return False
        # Allow more retrieval types including the default "similarity"
        valid_types = ["semantic", "keyword", "hybrid", "similarity"]
        if config.retrieval_type not in valid_types:
            return False
        return True
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the retriever's index."""
        self.document_store = documents
        
        # Prepare embeddings for semantic search
        texts = [doc.page_content for doc in documents]
        self.document_embeddings = self.embedding_model.encode(texts)
        
        # Prepare BM25 for keyword search
        self.bm25 = BM25Okapi([doc.page_content.split() for doc in documents])
    
    def retrieve(self, query: str, documents: List[Document]) -> List[Document]:
        """Retrieve documents using the specified method."""
        # Update document store if new documents provided
        if documents != self.document_store:
            self.add_documents(documents)
        
        if not self.document_store:
            return []
        
        # Choose retrieval method
        if self.config.retrieval_type == "semantic":
            results = self._semantic_search(query)
        elif self.config.retrieval_type == "keyword":
            results = self._keyword_search(query)
        else:  # hybrid
            results = self._hybrid_search(query)
        
        # Apply re-ranking if enabled
        if self.config.reranking_enabled and self.cross_encoder:
            results = self._rerank_documents(query, results)
        
        # Apply filtering
        filtered_results = self._filter_results(results)
        
        return [result['document'] for result in filtered_results[:self.config.top_k]]
    
    def _semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate cosine similarities
        similarities = np.dot(self.document_embeddings, query_embedding) / (
            np.linalg.norm(self.document_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Create results
        results = []
        for doc, score in zip(self.document_store, similarities):
            if score >= self.config.similarity_threshold:
                results.append({
                    'document': doc,
                    'score': float(score),
                    'retrieval_method': 'semantic'
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform keyword search using BM25."""
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Normalize scores to [0, 1]
        max_score = max(scores) if scores.any() else 1
        normalized_scores = scores / max_score if max_score > 0 else scores
        
        results = []
        for doc, score in zip(self.document_store, normalized_scores):
            if score >= self.config.similarity_threshold:
                results.append({
                    'document': doc,
                    'score': float(score),
                    'retrieval_method': 'keyword'
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def _hybrid_search(self, query: str) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results."""
        semantic_results = self._semantic_search(query)
        keyword_results = self._keyword_search(query)
        
        # Combine and normalize scores
        combined_results = {}
        for result in semantic_results + keyword_results:
            doc_id = id(result['document'])
            if doc_id not in combined_results:
                combined_results[doc_id] = result
            else:
                # Average scores if document appears in both results
                combined_results[doc_id]['score'] = (
                    combined_results[doc_id]['score'] + result['score']
                ) / 2
                combined_results[doc_id]['retrieval_method'] = 'hybrid'
        
        return sorted(list(combined_results.values()), key=lambda x: x['score'], reverse=True)
    
    def _rerank_documents(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank documents using cross-encoder."""
        if not results:  # Handle empty results list
            return results
            
        pairs = [[query, doc['document'].page_content] for doc in results]
        if not pairs:  # Double check pairs is not empty
            return results
            
        scores = self.cross_encoder.predict(pairs)
        
        reranked = []
        for score, result in zip(scores, results):
            reranked.append({
                'document': result['document'],
                'score': float(score),
                'retrieval_method': f"{result['retrieval_method']}_reranked"
            })
        
        return sorted(reranked, key=lambda x: x['score'], reverse=True)
    
    def _filter_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results based on threshold."""
        return [
            result for result in results 
            if result['score'] >= self.config.filtering_threshold
        ]