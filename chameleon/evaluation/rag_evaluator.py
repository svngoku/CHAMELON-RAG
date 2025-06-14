from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
import time
import uuid
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLM
from ragas import evaluate


class RAGEvaluator:
    """
    Comprehensive RAG evaluation system using LangChain and RAGAS metrics.
    
    Features:
    - Multiple evaluation metrics: relevance, faithfulness, correctness
    - Detailed per-query and overall evaluation scores
    - Support for ground truth comparisons when available
    - Hallucination detection
    
    References:
    - https://python.langchain.com/docs/guides/evaluation/
    - https://docs.ragas.io/en/latest/
    """
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        metrics: Optional[List[str]] = None,
        verbose: bool = False
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            llm: Language model for evaluation
            metrics: List of metrics to evaluate (default: all)
            verbose: Whether to print detailed evaluation info
        """
        self.llm = llm or ChatOpenAI(temperature=0)
        self.metrics = metrics or ["faithfulness", "answer_relevancy", "context_precision"]
        self.verbose = verbose
        self.evaluation_history = []
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        context_docs: List[Document],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG response.
        
        Args:
            query: The original query
            response: The generated response
            context_docs: The retrieved documents used for generation
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            Dictionary with evaluation scores and analysis
        """
        eval_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Prepare data for RAGAS evaluation
        contexts = [doc.page_content for doc in context_docs]
        
        # Select metrics to use
        metric_instances = []
        if "faithfulness" in self.metrics:
            metric_instances.append(faithfulness)
        if "answer_relevancy" in self.metrics:
            metric_instances.append(answer_relevancy)
        if "context_precision" in self.metrics:
            metric_instances.append(context_precision)
        if "context_recall" in self.metrics and ground_truth:
            metric_instances.append(context_recall)
        
        # Create dataset for evaluation
        eval_data = {
            "question": [query],
            "answer": [response],
            "contexts": [contexts],
        }
        
        if ground_truth:
            eval_data["ground_truth"] = [ground_truth]
        
        # Run evaluation with RAGAS
        try:
            ragas_llm = LangchainLLM(self.llm)
            result = evaluate(
                eval_data,
                metrics=metric_instances,
                llm=ragas_llm,
            )
            
            # Convert result to dictionary
            scores = {}
            for metric in result.keys():
                if metric != "metadata":
                    scores[metric] = float(result[metric][0])
            
            # Check for hallucination
            hallucination_score = 1.0 - scores.get("faithfulness", 1.0)
            hallucination_detected = hallucination_score > 0.3  # Threshold for hallucination
            
            # Generate overall score (weighted average)
            weights = {
                "faithfulness": 0.4,
                "answer_relevancy": 0.3,
                "context_precision": 0.2,
                "context_recall": 0.1
            }
            
            weighted_sum = 0
            weight_total = 0
            
            for metric, value in scores.items():
                if metric in weights:
                    weighted_sum += value * weights[metric]
                    weight_total += weights[metric]
            
            overall_score = weighted_sum / weight_total if weight_total > 0 else 0
            
            # Create evaluation result
            evaluation_result = {
                "id": eval_id,
                "timestamp": time.time(),
                "query": query,
                "response_length": len(response),
                "context_docs_count": len(context_docs),
                "evaluation_time_ms": int((time.time() - start_time) * 1000),
                "scores": scores,
                "overall_score": overall_score,
                "hallucination_detected": hallucination_detected,
                "hallucination_score": hallucination_score
            }
            
            # Store in history
            self.evaluation_history.append(evaluation_result)
            
            if self.verbose:
                print(f"Evaluation completed for query: {query}")
                print(f"Overall score: {overall_score:.2f}")
                print(f"Individual metrics: {scores}")
                if hallucination_detected:
                    print(f"WARNING: Potential hallucination detected (score: {hallucination_score:.2f})")
            
            return evaluation_result
            
        except Exception as e:
            error_result = {
                "id": eval_id,
                "timestamp": time.time(),
                "query": query,
                "error": str(e),
                "evaluation_time_ms": int((time.time() - start_time) * 1000),
            }
            
            if self.verbose:
                print(f"Evaluation error for query: {query}")
                print(f"Error: {str(e)}")
            
            return error_result
    
    def evaluate_batch(
        self,
        queries: List[str],
        responses: List[str],
        contexts: List[List[Document]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple RAG responses.
        
        Args:
            queries: List of queries
            responses: List of responses
            contexts: List of context document lists for each query
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dictionary with individual and aggregated evaluation results
        """
        results = []
        
        if ground_truths is None:
            ground_truths = [None] * len(queries)
        
        for i, (query, response, context, truth) in enumerate(zip(queries, responses, contexts, ground_truths)):
            result = self.evaluate_response(query, response, context, truth)
            results.append(result)
        
        # Calculate aggregate metrics
        agg_metrics = {}
        for metric in self.metrics:
            values = [r["scores"].get(metric, 0) for r in results if "scores" in r and metric in r["scores"]]
            if values:
                agg_metrics[metric] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        # Calculate aggregate overall score
        overall_scores = [r.get("overall_score", 0) for r in results if "overall_score" in r]
        overall_agg = {
            "mean": sum(overall_scores) / len(overall_scores) if overall_scores else 0,
            "min": min(overall_scores) if overall_scores else 0,
            "max": max(overall_scores) if overall_scores else 0
        }
        
        return {
            "individual_results": results,
            "aggregate_metrics": agg_metrics,
            "overall_score": overall_agg,
            "total_evaluated": len(results),
            "timestamp": time.time()
        }
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get the history of all evaluations performed."""
        return self.evaluation_history