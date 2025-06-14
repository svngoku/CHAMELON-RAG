"""
Health Checker for CHAMELEON RAG Framework
Monitors and validates component health during runtime.
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import traceback
import json
import os

from .logging_utils import setup_colored_logger, COLORS

@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Any
    status: str  # 'healthy', 'warning', 'critical'
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""

@dataclass
class ComponentHealth:
    """Health status of a component."""
    component_name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    metrics: List[HealthMetric] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    uptime: timedelta = field(default_factory=lambda: timedelta(0))

class ChameleonHealthChecker:
    """Health monitoring system for CHAMELEON RAG components."""
    
    def __init__(self, check_interval: int = 60):
        self.logger = setup_colored_logger()
        self.check_interval = check_interval
        self.components: Dict[str, ComponentHealth] = {}
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = datetime.now()
        
        # Health thresholds
        self.thresholds = {
            "memory_usage_percent": 80.0,
            "cpu_usage_percent": 80.0,
            "response_time_seconds": 10.0,
            "error_rate_percent": 5.0,
            "disk_usage_percent": 90.0
        }
        
        # Component checkers
        self.checkers: Dict[str, Callable] = {
            "system": self._check_system_health,
            "memory": self._check_memory_health,
            "pipeline": self._check_pipeline_health,
            "retriever": self._check_retriever_health,
            "generator": self._check_generator_health,
            "vector_store": self._check_vector_store_health
        }
    
    def register_component(self, component_name: str, component: Any):
        """Register a component for health monitoring."""
        self.components[component_name] = ComponentHealth(
            component_name=component_name,
            status="healthy"
        )
        self.logger.info(f"{COLORS['GREEN']}✅ Registered component: {component_name}{COLORS['ENDC']}")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring:
            self.logger.warning(f"{COLORS['YELLOW']}⚠️ Monitoring already started{COLORS['ENDC']}")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"{COLORS['BLUE']}🔍 Health monitoring started (interval: {self.check_interval}s){COLORS['ENDC']}")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info(f"{COLORS['BLUE']}⏹️ Health monitoring stopped{COLORS['ENDC']}")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self.check_all_components()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"{COLORS['RED']}❌ Error in monitoring loop: {str(e)}{COLORS['ENDC']}")
                time.sleep(self.check_interval)
    
    def check_all_components(self) -> Dict[str, ComponentHealth]:
        """Check health of all registered components."""
        for component_name in list(self.components.keys()):
            try:
                self._check_component_health(component_name)
            except Exception as e:
                self._record_component_error(component_name, str(e))
        
        return self.components
    
    def _check_component_health(self, component_name: str):
        """Check health of a specific component."""
        if component_name in self.checkers:
            checker = self.checkers[component_name]
            metrics = checker()
            
            # Update component health
            component = self.components[component_name]
            component.metrics = metrics
            component.last_check = datetime.now()
            component.uptime = datetime.now() - self.start_time
            
            # Determine overall status
            component.status = self._determine_component_status(metrics)
    
    def _determine_component_status(self, metrics: List[HealthMetric]) -> str:
        """Determine overall component status from metrics."""
        critical_count = sum(1 for m in metrics if m.status == 'critical')
        warning_count = sum(1 for m in metrics if m.status == 'warning')
        
        if critical_count > 0:
            return 'unhealthy'
        elif warning_count > 0:
            return 'degraded'
        else:
            return 'healthy'
    
    def _record_component_error(self, component_name: str, error: str):
        """Record an error for a component."""
        if component_name in self.components:
            self.components[component_name].error_count += 1
            self.components[component_name].status = 'unhealthy'
            self.logger.error(f"{COLORS['RED']}❌ {component_name} error: {error}{COLORS['ENDC']}")
    
    def _check_system_health(self) -> List[HealthMetric]:
        """Check system-level health metrics."""
        metrics = []
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(HealthMetric(
                name="cpu_usage_percent",
                value=cpu_percent,
                status="critical" if cpu_percent > self.thresholds["cpu_usage_percent"] else
                       "warning" if cpu_percent > self.thresholds["cpu_usage_percent"] * 0.8 else "healthy",
                threshold=self.thresholds["cpu_usage_percent"],
                message=f"CPU usage: {cpu_percent:.1f}%"
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.append(HealthMetric(
                name="memory_usage_percent",
                value=memory.percent,
                status="critical" if memory.percent > self.thresholds["memory_usage_percent"] else
                       "warning" if memory.percent > self.thresholds["memory_usage_percent"] * 0.8 else "healthy",
                threshold=self.thresholds["memory_usage_percent"],
                message=f"Memory usage: {memory.percent:.1f}%"
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(HealthMetric(
                name="disk_usage_percent",
                value=disk_percent,
                status="critical" if disk_percent > self.thresholds["disk_usage_percent"] else
                       "warning" if disk_percent > self.thresholds["disk_usage_percent"] * 0.8 else "healthy",
                threshold=self.thresholds["disk_usage_percent"],
                message=f"Disk usage: {disk_percent:.1f}%"
            ))
            
        except Exception as e:
            metrics.append(HealthMetric(
                name="system_check_error",
                value=str(e),
                status="critical",
                message=f"System check failed: {str(e)}"
            ))
        
        return metrics
    
    def _check_memory_health(self) -> List[HealthMetric]:
        """Check memory-specific health metrics."""
        metrics = []
        
        try:
            import gc
            
            # Garbage collection stats
            gc_stats = gc.get_stats()
            metrics.append(HealthMetric(
                name="gc_collections",
                value=len(gc_stats),
                status="healthy",
                message=f"GC generations: {len(gc_stats)}"
            ))
            
            # Object count
            obj_count = len(gc.get_objects())
            metrics.append(HealthMetric(
                name="object_count",
                value=obj_count,
                status="warning" if obj_count > 100000 else "healthy",
                message=f"Objects in memory: {obj_count:,}"
            ))
            
        except Exception as e:
            metrics.append(HealthMetric(
                name="memory_check_error",
                value=str(e),
                status="critical",
                message=f"Memory check failed: {str(e)}"
            ))
        
        return metrics
    
    def _check_pipeline_health(self) -> List[HealthMetric]:
        """Check RAG pipeline health."""
        metrics = []
        
        # This would be implemented based on actual pipeline monitoring
        # For now, return basic health status
        metrics.append(HealthMetric(
            name="pipeline_status",
            value="operational",
            status="healthy",
            message="Pipeline is operational"
        ))
        
        return metrics
    
    def _check_retriever_health(self) -> List[HealthMetric]:
        """Check retriever component health."""
        metrics = []
        
        # Placeholder for retriever-specific checks
        metrics.append(HealthMetric(
            name="retriever_status",
            value="operational",
            status="healthy",
            message="Retriever is operational"
        ))
        
        return metrics
    
    def _check_generator_health(self) -> List[HealthMetric]:
        """Check generator component health."""
        metrics = []
        
        # Placeholder for generator-specific checks
        metrics.append(HealthMetric(
            name="generator_status",
            value="operational",
            status="healthy",
            message="Generator is operational"
        ))
        
        return metrics
    
    def _check_vector_store_health(self) -> List[HealthMetric]:
        """Check vector store health."""
        metrics = []
        
        # Placeholder for vector store-specific checks
        metrics.append(HealthMetric(
            name="vector_store_status",
            value="operational",
            status="healthy",
            message="Vector store is operational"
        ))
        
        return metrics
    
    def _check_api_keys(self) -> Dict[str, Any]:
        """Check if required API keys are available."""
        api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
            "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
            "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),  # For LiteLLM
        }
        
        available_keys = {k: bool(v) for k, v in api_keys.items()}
        total_keys = len(api_keys)
        available_count = sum(available_keys.values())
        
        return {
            "status": "healthy" if available_count > 0 else "warning",
            "available_keys": available_keys,
            "summary": f"{available_count}/{total_keys} API keys configured",
            "details": "At least one API key should be configured for LLM access"
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        try:
            system_health = self._check_system_health()
            memory_health = self._check_memory_health()
            disk_health = self._check_disk_health()
            api_keys_health = self._check_api_keys()
            
            # Overall status
            statuses = [
                system_health["status"],
                memory_health["status"], 
                disk_health["status"],
                api_keys_health["status"]
            ]
            
            if "critical" in statuses:
                overall_status = "critical"
            elif "warning" in statuses:
                overall_status = "warning"
            else:
                overall_status = "healthy"
            
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": overall_status,
                "components": {
                    "system": system_health,
                    "memory": memory_health,
                    "disk": disk_health,
                    "api_keys": api_keys_health
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting health summary: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "critical",
                "error": str(e)
            }
    
    def print_health_report(self):
        """Print a comprehensive health report."""
        summary = self.get_health_summary()
        
        print(f"\n{COLORS['BLUE']}🏥 CHAMELEON HEALTH REPORT{COLORS['ENDC']}")
        print("=" * 60)
        
        # Overall status
        status_color = COLORS['GREEN'] if summary['overall_status'] == 'healthy' else \
                      COLORS['YELLOW'] if summary['overall_status'] == 'degraded' else COLORS['RED']
        print(f"Overall Status: {status_color}{summary['overall_status'].upper()}{COLORS['ENDC']}")
        print(f"Uptime: {summary['uptime']}")
        print(f"Last Check: {summary['last_check'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Component summary
        print(f"\n{COLORS['BLUE']}📊 Component Summary:{COLORS['ENDC']}")
        print(f"  • Total: {summary['total_components']}")
        print(f"  • {COLORS['GREEN']}Healthy: {summary['healthy_components']}{COLORS['ENDC']}")
        print(f"  • {COLORS['YELLOW']}Degraded: {summary['degraded_components']}{COLORS['ENDC']}")
        print(f"  • {COLORS['RED']}Unhealthy: {summary['unhealthy_components']}{COLORS['ENDC']}")
        
        # Detailed component status
        print(f"\n{COLORS['BLUE']}🔍 Component Details:{COLORS['ENDC']}")
        for name, component in self.components.items():
            status_color = COLORS['GREEN'] if component.status == 'healthy' else \
                          COLORS['YELLOW'] if component.status == 'degraded' else COLORS['RED']
            
            print(f"\n{status_color}● {name.upper()}{COLORS['ENDC']} ({component.status})")
            print(f"  Last Check: {component.last_check.strftime('%H:%M:%S')}")
            print(f"  Errors: {component.error_count}")
            
            # Show critical/warning metrics
            critical_metrics = [m for m in component.metrics if m.status in ['critical', 'warning']]
            if critical_metrics:
                for metric in critical_metrics:
                    metric_color = COLORS['RED'] if metric.status == 'critical' else COLORS['YELLOW']
                    print(f"  {metric_color}⚠️ {metric.message}{COLORS['ENDC']}")
    
    def export_health_data(self, filepath: str):
        """Export health data to JSON file."""
        health_data = {
            "summary": self.get_health_summary(),
            "components": {}
        }
        
        for name, component in self.components.items():
            health_data["components"][name] = {
                "status": component.status,
                "last_check": component.last_check.isoformat(),
                "error_count": component.error_count,
                "uptime": str(component.uptime),
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "status": m.status,
                        "threshold": m.threshold,
                        "timestamp": m.timestamp.isoformat(),
                        "message": m.message
                    }
                    for m in component.metrics
                ]
            }
        
        # Convert timedelta to string for JSON serialization
        health_data["summary"]["uptime"] = str(health_data["summary"]["uptime"])
        health_data["summary"]["last_check"] = health_data["summary"]["last_check"].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(health_data, f, indent=2)
        
        self.logger.info(f"{COLORS['GREEN']}💾 Health data exported to {filepath}{COLORS['ENDC']}")

# Global health checker instance
_health_checker = None

def get_health_checker() -> ChameleonHealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = ChameleonHealthChecker()
    return _health_checker

def start_health_monitoring(check_interval: int = 60):
    """Start global health monitoring."""
    checker = get_health_checker()
    checker.check_interval = check_interval
    checker.start_monitoring()

def stop_health_monitoring():
    """Stop global health monitoring."""
    checker = get_health_checker()
    checker.stop_monitoring()

def register_component(component_name: str, component: Any):
    """Register a component for health monitoring."""
    checker = get_health_checker()
    checker.register_component(component_name, component)

def get_health_summary() -> Dict[str, Any]:
    """Get current health summary."""
    checker = get_health_checker()
    return checker.get_health_summary()

def print_health_report():
    """Print current health report."""
    checker = get_health_checker()
    checker.print_health_report()

if __name__ == "__main__":
    # Demo the health checker
    checker = ChameleonHealthChecker()
    checker.register_component("system", None)
    checker.register_component("memory", None)
    checker.check_all_components()
    checker.print_health_report() 