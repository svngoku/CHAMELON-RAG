"""
Comprehensive Test Framework for CHAMELEON RAG
Tests all components and validates proper connections.
"""

import sys
import time
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain_core.documents import Document

from .logging_utils import setup_colored_logger, COLORS
from .structure_validator import validate_chameleon_structure
from .pipeline_builder import ChameleonPipelineBuilder, quick_pipeline
from .health_checker import get_health_checker

@dataclass
class TestResult:
    """Result of a test case."""
    test_name: str
    status: str  # 'pass', 'fail', 'skip'
    message: str
    duration: float = 0.0
    error: Optional[str] = None

class ChameleonTestFramework:
    """Comprehensive test framework for CHAMELEON RAG."""
    
    def __init__(self):
        self.logger = setup_colored_logger()
        self.results: List[TestResult] = []
        
        # Test documents
        self.test_documents = [
            Document(
                page_content="CHAMELEON is a flexible RAG framework that supports multiple pipeline types.",
                metadata={"source": "test_doc_1", "type": "framework"}
            ),
            Document(
                page_content="The framework includes advanced retrieval, generation, and memory components.",
                metadata={"source": "test_doc_2", "type": "components"}
            ),
            Document(
                page_content="Users can easily build custom RAG pipelines using the builder pattern.",
                metadata={"source": "test_doc_3", "type": "usage"}
            )
        ]
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites."""
        self.logger.info(f"{COLORS['BLUE']}🧪 Starting CHAMELEON comprehensive tests...{COLORS['ENDC']}")
        
        # Clear previous results
        self.results = []
        
        # Run test suites
        self._test_structure_validation()
        self._test_imports()
        self._test_pipeline_builder()
        self._test_basic_pipeline()
        self._test_enhanced_pipeline()
        self._test_health_monitoring()
        self._test_vector_store_creation()
        self._test_memory_components()
        
        # Generate report
        return self._generate_test_report()
    
    def _run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        self.logger.info(f"{COLORS['BLUE']}Running: {test_name}{COLORS['ENDC']}")
        
        start_time = time.time()
        try:
            test_func()
            duration = time.time() - start_time
            self.results.append(TestResult(
                test_name=test_name,
                status="pass",
                message="Test passed successfully",
                duration=duration
            ))
            self.logger.info(f"{COLORS['GREEN']}✅ {test_name} - PASSED ({duration:.2f}s){COLORS['ENDC']}")
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            self.results.append(TestResult(
                test_name=test_name,
                status="fail",
                message=f"Test failed: {error_msg}",
                duration=duration,
                error=traceback.format_exc()
            ))
            self.logger.error(f"{COLORS['RED']}❌ {test_name} - FAILED ({duration:.2f}s): {error_msg}{COLORS['ENDC']}")
    
    def _test_structure_validation(self):
        """Test structure validation."""
        def test_structure():
            report = validate_chameleon_structure()
            if report['summary']['failed'] > 0:
                raise AssertionError(f"Structure validation failed: {report['summary']['failed']} issues")
        
        self._run_test("Structure Validation", test_structure)
    
    def _test_imports(self):
        """Test critical imports."""
        def test_imports():
            # Test base imports
            from chameleon.base import (
                BaseRetriever, BaseGenerator, BaseMemory,
                RetrieverConfig, GeneratorConfig, MemoryConfig, PipelineConfig
            )
            
            # Test pipeline imports
            from chameleon.pipeline.rag_pipeline import RAGPipeline
            from chameleon.pipeline.enhanced_rag_pipeline import EnhancedRAGPipeline
            
            # Test component imports
            from chameleon.retrieval.advanced_retriever import AdvancedRetriever
            from chameleon.generation.llm_generator import LLMGenerator
            from chameleon.memory.entity_memory import EntityMemory
            
            # Test utility imports
            from chameleon.utils import (
                ChameleonPipelineBuilder, validate_chameleon_structure,
                get_health_checker
            )
        
        self._run_test("Critical Imports", test_imports)
    
    def _test_pipeline_builder(self):
        """Test pipeline builder functionality."""
        def test_builder():
            builder = ChameleonPipelineBuilder()
            
            # Test template listing
            builder.list_templates()
            
            # Test builder configuration
            builder.from_template("quick_start")
            builder.with_documents(self.test_documents)
            builder.with_openai("gpt-3.5-turbo")
            
            # Test configuration validation
            issues = builder.validate_configuration()
            if len(issues) > 2:  # Allow some minor issues
                raise AssertionError(f"Too many configuration issues: {issues}")
            
            # Test configuration summary
            summary = builder.get_configuration_summary()
            assert summary['documents_count'] == len(self.test_documents)
            assert summary['rag_type'] == 'modular'
        
        self._run_test("Pipeline Builder", test_builder)
    
    def _test_basic_pipeline(self):
        """Test basic pipeline creation and execution."""
        def test_basic():
            # Skip if no API key
            import os
            if not os.getenv('OPENAI_API_KEY'):
                raise Exception("OPENAI_API_KEY not set - skipping pipeline test")
            
            # Create basic pipeline
            pipeline = quick_pipeline(
                documents=self.test_documents,
                model_name="gpt-3.5-turbo",
                provider="openai"
            )
            
            # Test pipeline attributes
            assert hasattr(pipeline, 'run')
            assert hasattr(pipeline, 'title')
            assert len(pipeline.documents) == len(self.test_documents)
            
            # Test simple query (with timeout)
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Pipeline test timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout
            
            try:
                response = pipeline.run("What is CHAMELEON?")
                assert 'response' in response
                assert len(response['response']) > 0
            finally:
                signal.alarm(0)  # Cancel timeout
        
        self._run_test("Basic Pipeline", test_basic)
    
    def _test_enhanced_pipeline(self):
        """Test enhanced pipeline features."""
        def test_enhanced():
            # Skip if no API key
            import os
            if not os.getenv('OPENAI_API_KEY'):
                raise Exception("OPENAI_API_KEY not set - skipping enhanced pipeline test")
            
            from chameleon.pipeline.enhanced_rag_pipeline import EnhancedRAGPipeline
            from chameleon.base import PipelineConfig, RetrieverConfig, GeneratorConfig, MemoryConfig
            
            # Create enhanced pipeline
            config = PipelineConfig(
                rag_type="enhanced",
                retriever_config=RetrieverConfig(
                    top_k=3,
                    similarity_threshold=0.5,
                    retrieval_type="semantic"
                ),
                generator_config=GeneratorConfig(
                    model_name="gpt-3.5-turbo",
                    provider="openai",
                    temperature=0.7,
                    max_tokens=500
                ),
                memory_config=MemoryConfig(
                    memory_type="buffer",
                    max_history=5
                )
            )
            
            pipeline = EnhancedRAGPipeline(
                title="Test Enhanced Pipeline",
                documents=self.test_documents,
                config=config,
                enable_evaluation=False
            )
            
            # Test pipeline attributes
            assert hasattr(pipeline, 'run')
            assert hasattr(pipeline, 'get_stats')
            
            # Test stats
            stats = pipeline.get_stats()
            assert 'title' in stats
            assert 'document_count' in stats
        
        self._run_test("Enhanced Pipeline", test_enhanced)
    
    def _test_health_monitoring(self):
        """Test health monitoring system."""
        def test_health():
            checker = get_health_checker()
            
            # Register test components
            checker.register_component("test_system", None)
            checker.register_component("test_memory", None)
            
            # Check components
            results = checker.check_all_components()
            assert len(results) >= 2
            
            # Get health summary
            summary = checker.get_health_summary()
            assert 'overall_status' in summary
            assert 'total_components' in summary
            
            # Test health report
            checker.print_health_report()
        
        self._run_test("Health Monitoring", test_health)
    
    def _test_vector_store_creation(self):
        """Test vector store creation with batching."""
        def test_vector_store():
            # Skip if no API key
            import os
            if not os.getenv('OPENAI_API_KEY'):
                raise Exception("OPENAI_API_KEY not set - skipping vector store test")
            
            from chameleon.vector_db_factory import VectorDBFactory, VectorDBConfig
            
            # Create vector store config
            config = VectorDBConfig(
                store_type="faiss",
                embedding_model="text-embedding-3-small",
                chunk_size=500,
                chunk_overlap=50
            )
            
            # Create factory
            factory = VectorDBFactory(config=config)
            
            # Test vector store creation
            vector_store = factory.create_vectorstore(self.test_documents)
            assert vector_store is not None
            
            # Test similarity search
            results = vector_store.similarity_search("CHAMELEON framework", k=2)
            assert len(results) <= 2
        
        self._run_test("Vector Store Creation", test_vector_store)
    
    def _test_memory_components(self):
        """Test memory components."""
        def test_memory():
            from chameleon.memory.entity_memory import EntityMemory
            from chameleon.base import MemoryConfig
            
            # Create memory config
            config = MemoryConfig(
                memory_type="entity",
                max_history=10
            )
            
            # Create entity memory
            memory = EntityMemory(config)
            
            # Test memory operations
            test_query = "What is CHAMELEON?"
            test_response = {"response": "CHAMELEON is a RAG framework"}
            
            memory.add(test_query, test_response)
            
            # Test memory retrieval
            history = memory.get()
            assert len(history) >= 1
            
            # Test entity extraction
            entities = memory.extract_entities(test_query)
            assert isinstance(entities, list)
        
        self._run_test("Memory Components", test_memory)
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        passed = [r for r in self.results if r.status == "pass"]
        failed = [r for r in self.results if r.status == "fail"]
        skipped = [r for r in self.results if r.status == "skip"]
        
        total_duration = sum(r.duration for r in self.results)
        
        report = {
            "summary": {
                "total_tests": len(self.results),
                "passed": len(passed),
                "failed": len(failed),
                "skipped": len(skipped),
                "success_rate": len(passed) / len(self.results) * 100 if self.results else 0,
                "total_duration": total_duration
            },
            "results": {
                "passed": passed,
                "failed": failed,
                "skipped": skipped
            }
        }
        
        # Print summary
        self.logger.info(f"\n{COLORS['BLUE']}📊 TEST SUMMARY{COLORS['ENDC']}")
        self.logger.info("=" * 50)
        self.logger.info(f"Total tests: {report['summary']['total_tests']}")
        self.logger.info(f"{COLORS['GREEN']}✅ Passed: {report['summary']['passed']}{COLORS['ENDC']}")
        self.logger.info(f"{COLORS['RED']}❌ Failed: {report['summary']['failed']}{COLORS['ENDC']}")
        self.logger.info(f"{COLORS['YELLOW']}⏭️ Skipped: {report['summary']['skipped']}{COLORS['ENDC']}")
        self.logger.info(f"Success rate: {report['summary']['success_rate']:.1f}%")
        self.logger.info(f"Total duration: {report['summary']['total_duration']:.2f}s")
        
        # Print failed tests
        if failed:
            self.logger.info(f"\n{COLORS['RED']}❌ FAILED TESTS:{COLORS['ENDC']}")
            for result in failed:
                self.logger.info(f"  • {result.test_name}: {result.message}")
        
        # Overall status
        if len(failed) == 0:
            self.logger.info(f"\n{COLORS['GREEN']}🎉 ALL TESTS PASSED! Framework is ready to use.{COLORS['ENDC']}")
        elif len(failed) <= 2:
            self.logger.info(f"\n{COLORS['YELLOW']}⚠️ Minor issues found. Framework is mostly functional.{COLORS['ENDC']}")
        else:
            self.logger.info(f"\n{COLORS['RED']}🚨 Multiple test failures. Please review the framework setup.{COLORS['ENDC']}")
        
        return report

def run_comprehensive_tests() -> Dict[str, Any]:
    """Run comprehensive tests on the CHAMELEON framework."""
    framework = ChameleonTestFramework()
    return framework.run_all_tests()

if __name__ == "__main__":
    # Run tests when script is executed directly
    run_comprehensive_tests() 