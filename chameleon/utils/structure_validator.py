"""
Project Structure Validator for CHAMELEON RAG Framework
Validates imports, dependencies, and overall project structure.
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import ast
import traceback

from .logging_utils import setup_colored_logger, COLORS

@dataclass
class ValidationResult:
    """Result of a validation check."""
    component: str
    status: str  # 'pass', 'fail', 'warning'
    message: str
    details: Optional[str] = None

class ChameleonStructureValidator:
    """Validates the CHAMELEON RAG framework structure and connections."""
    
    def __init__(self, project_root: Optional[str] = None):
        self.logger = setup_colored_logger()
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.chameleon_root = self.project_root / "chameleon"
        self.results: List[ValidationResult] = []
        
        # Expected project structure
        self.expected_structure = {
            "chameleon": {
                "base.py": "Core base classes and configurations",
                "vector_db_factory.py": "Vector database factory",
                "__init__.py": "Package initialization",
                "pipeline/": {
                    "rag_pipeline.py": "Basic RAG pipeline",
                    "enhanced_rag_pipeline.py": "Enhanced RAG pipeline",
                    "__init__.py": "Pipeline package init"
                },
                "retrieval/": {
                    "advanced_retriever.py": "Advanced retrieval methods",
                    "simple_retriever.py": "Simple retrieval methods",
                    "multi_query_retriever.py": "Multi-query retrieval",
                    "parent_document_retriever.py": "Parent document retrieval",
                    "__init__.py": "Retrieval package init"
                },
                "generation/": {
                    "llm_generator.py": "LLM-based generation",
                    "advanced_generator.py": "Advanced generation methods",
                    "__init__.py": "Generation package init"
                },
                "memory/": {
                    "entity_memory.py": "Entity-based memory",
                    "memory_factory.py": "Memory factory",
                    "memory_adapter.py": "Memory adapters",
                    "__init__.py": "Memory package init"
                },
                "preprocessing/": {
                    "query_transformer.py": "Query transformation",
                    "markdown_chunking.py": "Markdown processing",
                    "__init__.py": "Preprocessing package init"
                },
                "postprocessing/": {
                    "contextual_compressor.py": "Context compression",
                    "__init__.py": "Postprocessing package init"
                },
                "utils/": {
                    "logging_utils.py": "Logging utilities",
                    "utils.py": "General utilities",
                    "__init__.py": "Utils package init"
                },
                "evaluation/": {
                    "rag_evaluator.py": "RAG evaluation methods",
                    "__init__.py": "Evaluation package init"
                },
                "tools/": {
                    "__init__.py": "Tools package init"
                },
                "loaders/": {
                    "__init__.py": "Loaders package init"
                }
            }
        }
        
        # Core dependencies to check
        self.core_dependencies = [
            "langchain",
            "langchain_openai", 
            "langchain_together",
            "langchain_community",
            "langchain_core",
            "faiss-cpu",
            "sentence_transformers",
            "openai",
            "together",
            "pydantic",
            "numpy",
            "datasets"
        ]
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        self.logger.info(f"{COLORS['BLUE']}🔍 Starting CHAMELEON structure validation...{COLORS['ENDC']}")
        
        # Clear previous results
        self.results = []
        
        # Run validation checks
        self._validate_project_structure()
        self._validate_imports()
        self._validate_base_classes()
        self._validate_pipeline_connections()
        self._validate_dependencies()
        self._validate_examples()
        
        # Generate report
        return self._generate_report()
    
    def _validate_project_structure(self):
        """Validate the project directory structure."""
        self.logger.info(f"{COLORS['BLUE']}📁 Validating project structure...{COLORS['ENDC']}")
        
        def check_structure(expected: Dict, current_path: Path, prefix: str = ""):
            for name, description in expected.items():
                full_path = current_path / name
                
                if isinstance(description, dict):
                    # It's a directory
                    if full_path.is_dir():
                        self.results.append(ValidationResult(
                            f"{prefix}{name}/",
                            "pass",
                            f"Directory exists: {full_path}"
                        ))
                        check_structure(description, full_path, f"{prefix}{name}/")
                    else:
                        self.results.append(ValidationResult(
                            f"{prefix}{name}/",
                            "fail",
                            f"Missing directory: {full_path}"
                        ))
                else:
                    # It's a file
                    if full_path.is_file():
                        self.results.append(ValidationResult(
                            f"{prefix}{name}",
                            "pass",
                            f"File exists: {description}"
                        ))
                    else:
                        self.results.append(ValidationResult(
                            f"{prefix}{name}",
                            "fail",
                            f"Missing file: {description}"
                        ))
        
        check_structure(self.expected_structure, self.project_root)
    
    def _validate_imports(self):
        """Validate that all modules can be imported."""
        self.logger.info(f"{COLORS['BLUE']}📦 Validating imports...{COLORS['ENDC']}")
        
        # Key modules to test
        key_modules = [
            "chameleon.base",
            "chameleon.pipeline.rag_pipeline",
            "chameleon.pipeline.enhanced_rag_pipeline",
            "chameleon.retrieval.advanced_retriever",
            "chameleon.generation.llm_generator",
            "chameleon.memory.entity_memory",
            "chameleon.utils.logging_utils",
            "chameleon.vector_db_factory"
        ]
        
        for module_name in key_modules:
            try:
                module = importlib.import_module(module_name)
                self.results.append(ValidationResult(
                    f"import {module_name}",
                    "pass",
                    f"Successfully imported {module_name}"
                ))
            except ImportError as e:
                self.results.append(ValidationResult(
                    f"import {module_name}",
                    "fail",
                    f"Failed to import {module_name}: {str(e)}"
                ))
            except Exception as e:
                self.results.append(ValidationResult(
                    f"import {module_name}",
                    "warning",
                    f"Import warning for {module_name}: {str(e)}"
                ))
    
    def _validate_base_classes(self):
        """Validate that base classes are properly defined."""
        self.logger.info(f"{COLORS['BLUE']}🏗️ Validating base classes...{COLORS['ENDC']}")
        
        try:
            from chameleon.base import (
                BaseRetriever, BaseGenerator, BaseMemory, 
                BasePreprocessor, BasePostprocessor,
                RetrieverConfig, GeneratorConfig, MemoryConfig, PipelineConfig
            )
            
            # Check base classes
            base_classes = [
                (BaseRetriever, "BaseRetriever"),
                (BaseGenerator, "BaseGenerator"), 
                (BaseMemory, "BaseMemory"),
                (BasePreprocessor, "BasePreprocessor"),
                (BasePostprocessor, "BasePostprocessor")
            ]
            
            for base_class, name in base_classes:
                if inspect.isclass(base_class) and hasattr(base_class, '__abstractmethods__'):
                    self.results.append(ValidationResult(
                        f"base_class_{name}",
                        "pass",
                        f"{name} is properly defined as abstract base class"
                    ))
                else:
                    self.results.append(ValidationResult(
                        f"base_class_{name}",
                        "warning",
                        f"{name} may not be properly defined as abstract base class"
                    ))
            
            # Check config classes
            config_classes = [
                (RetrieverConfig, "RetrieverConfig"),
                (GeneratorConfig, "GeneratorConfig"),
                (MemoryConfig, "MemoryConfig"),
                (PipelineConfig, "PipelineConfig")
            ]
            
            for config_class, name in config_classes:
                if hasattr(config_class, '__annotations__'):
                    self.results.append(ValidationResult(
                        f"config_{name}",
                        "pass",
                        f"{name} is properly defined with type annotations"
                    ))
                else:
                    self.results.append(ValidationResult(
                        f"config_{name}",
                        "fail",
                        f"{name} missing type annotations"
                    ))
                    
        except ImportError as e:
            self.results.append(ValidationResult(
                "base_classes",
                "fail",
                f"Failed to import base classes: {str(e)}"
            ))
    
    def _validate_pipeline_connections(self):
        """Validate that pipeline components are properly connected."""
        self.logger.info(f"{COLORS['BLUE']}🔗 Validating pipeline connections...{COLORS['ENDC']}")
        
        try:
            # Test basic pipeline creation
            from chameleon.pipeline.rag_pipeline import RAGPipeline
            from chameleon.base import PipelineConfig
            from langchain_core.documents import Document
            
            # Create minimal test documents
            test_docs = [Document(page_content="Test content", metadata={"source": "test"})]
            
            # Test pipeline initialization
            config = PipelineConfig()
            pipeline = RAGPipeline(
                title="Test Pipeline",
                documents=test_docs,
                config=config
            )
            
            self.results.append(ValidationResult(
                "pipeline_creation",
                "pass",
                "RAGPipeline can be instantiated successfully"
            ))
            
            # Test pipeline methods
            if hasattr(pipeline, 'run'):
                self.results.append(ValidationResult(
                    "pipeline_run_method",
                    "pass",
                    "Pipeline has run method"
                ))
            else:
                self.results.append(ValidationResult(
                    "pipeline_run_method",
                    "fail",
                    "Pipeline missing run method"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                "pipeline_connections",
                "fail",
                f"Pipeline connection test failed: {str(e)}",
                traceback.format_exc()
            ))
    
    def _validate_dependencies(self):
        """Validate that all required dependencies are available."""
        self.logger.info(f"{COLORS['BLUE']}📋 Validating dependencies...{COLORS['ENDC']}")
        
        for dep in self.core_dependencies:
            try:
                # Handle special cases
                if dep == "faiss-cpu":
                    import faiss
                    dep_name = "faiss"
                elif dep == "sentence_transformers":
                    import sentence_transformers
                    dep_name = "sentence_transformers"
                else:
                    module = importlib.import_module(dep.replace("-", "_"))
                    dep_name = dep
                
                self.results.append(ValidationResult(
                    f"dependency_{dep_name}",
                    "pass",
                    f"Dependency {dep} is available"
                ))
                
            except ImportError:
                self.results.append(ValidationResult(
                    f"dependency_{dep}",
                    "fail",
                    f"Missing dependency: {dep}"
                ))
    
    def _validate_examples(self):
        """Validate that example files are present and syntactically correct."""
        self.logger.info(f"{COLORS['BLUE']}📝 Validating examples...{COLORS['ENDC']}")
        
        examples_dir = self.project_root / "examples"
        if not examples_dir.exists():
            self.results.append(ValidationResult(
                "examples_directory",
                "fail",
                "Examples directory not found"
            ))
            return
        
        # Check for key example files
        expected_examples = [
            "rag_example.py",
            "enhanced_rag_example.py"
        ]
        
        for example_file in expected_examples:
            example_path = examples_dir / example_file
            if example_path.exists():
                # Check syntax
                try:
                    with open(example_path, 'r') as f:
                        content = f.read()
                    ast.parse(content)
                    self.results.append(ValidationResult(
                        f"example_{example_file}",
                        "pass",
                        f"Example {example_file} exists and has valid syntax"
                    ))
                except SyntaxError as e:
                    self.results.append(ValidationResult(
                        f"example_{example_file}",
                        "fail",
                        f"Syntax error in {example_file}: {str(e)}"
                    ))
            else:
                self.results.append(ValidationResult(
                    f"example_{example_file}",
                    "warning",
                    f"Example file {example_file} not found"
                ))
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        passed = [r for r in self.results if r.status == "pass"]
        failed = [r for r in self.results if r.status == "fail"]
        warnings = [r for r in self.results if r.status == "warning"]
        
        report = {
            "summary": {
                "total_checks": len(self.results),
                "passed": len(passed),
                "failed": len(failed),
                "warnings": len(warnings),
                "success_rate": len(passed) / len(self.results) * 100 if self.results else 0
            },
            "results": {
                "passed": passed,
                "failed": failed,
                "warnings": warnings
            }
        }
        
        # Print summary
        self.logger.info(f"\n{COLORS['GREEN']}📊 VALIDATION SUMMARY{COLORS['ENDC']}")
        self.logger.info(f"Total checks: {report['summary']['total_checks']}")
        self.logger.info(f"{COLORS['GREEN']}✅ Passed: {report['summary']['passed']}{COLORS['ENDC']}")
        self.logger.info(f"{COLORS['RED']}❌ Failed: {report['summary']['failed']}{COLORS['ENDC']}")
        self.logger.info(f"{COLORS['YELLOW']}⚠️ Warnings: {report['summary']['warnings']}{COLORS['ENDC']}")
        self.logger.info(f"Success rate: {report['summary']['success_rate']:.1f}%")
        
        # Print failed checks
        if failed:
            self.logger.info(f"\n{COLORS['RED']}❌ FAILED CHECKS:{COLORS['ENDC']}")
            for result in failed:
                self.logger.info(f"  • {result.component}: {result.message}")
        
        # Print warnings
        if warnings:
            self.logger.info(f"\n{COLORS['YELLOW']}⚠️ WARNINGS:{COLORS['ENDC']}")
            for result in warnings:
                self.logger.info(f"  • {result.component}: {result.message}")
        
        return report

def validate_chameleon_structure(project_root: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to validate CHAMELEON structure."""
    validator = ChameleonStructureValidator(project_root)
    return validator.validate_all()

if __name__ == "__main__":
    # Run validation when script is executed directly
    validate_chameleon_structure() 