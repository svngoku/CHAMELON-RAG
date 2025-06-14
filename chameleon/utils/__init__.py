"""
CHAMELEON RAG Framework Utilities
Comprehensive utilities for building, validating, and monitoring RAG pipelines.
"""

from .logging_utils import setup_colored_logger, COLORS
from .structure_validator import (
    ChameleonStructureValidator,
    ValidationResult,
    validate_chameleon_structure
)
from .pipeline_builder import (
    ChameleonPipelineBuilder,
    PipelineTemplate,
    RAGType,
    Provider,
    VectorStore,
    quick_pipeline,
    production_pipeline,
    research_pipeline
)
from .health_checker import (
    ChameleonHealthChecker,
    HealthMetric,
    ComponentHealth,
    get_health_checker,
    start_health_monitoring,
    stop_health_monitoring,
    register_component,
    get_health_summary,
    print_health_report
)

# Import existing utils
from .utils import *

__all__ = [
    # Logging
    "setup_colored_logger",
    "COLORS",
    
    # Structure Validation
    "ChameleonStructureValidator",
    "ValidationResult", 
    "validate_chameleon_structure",
    
    # Pipeline Building
    "ChameleonPipelineBuilder",
    "PipelineTemplate",
    "RAGType",
    "Provider", 
    "VectorStore",
    "quick_pipeline",
    "production_pipeline",
    "research_pipeline",
    
    # Health Monitoring
    "ChameleonHealthChecker",
    "HealthMetric",
    "ComponentHealth",
    "get_health_checker",
    "start_health_monitoring",
    "stop_health_monitoring", 
    "register_component",
    "get_health_summary",
    "print_health_report",
]
