import time
import psutil
import threading
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json
import os
from collections import defaultdict
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """
    Real-time performance profiler for Brein AI system monitoring.
    """

    def __init__(self, log_interval: float = 5.0):
        self.log_interval = log_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
        self.performance_logs = "performance_logs/"
        os.makedirs(self.performance_logs, exist_ok=True)

        # Performance metrics
        self.metrics = {
            "cpu_percent": [],
            "memory_percent": [],
            "memory_mb": [],
            "disk_usage": [],
            "network_connections": [],
            "active_threads": [],
            "timestamp": []
        }

        # Operation-specific metrics
        self.operation_metrics = defaultdict(list)

    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(self.log_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.log_interval)

    def _collect_system_metrics(self):
        """Collect current system performance metrics."""
        timestamp = datetime.now().isoformat()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / (1024 * 1024)

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # Network connections
        network_connections = len(psutil.net_connections())

        # Active threads
        active_threads = threading.active_count()

        # Store metrics
        self.metrics["cpu_percent"].append(cpu_percent)
        self.metrics["memory_percent"].append(memory_percent)
        self.metrics["memory_mb"].append(memory_mb)
        self.metrics["disk_usage"].append(disk_percent)
        self.metrics["network_connections"].append(network_connections)
        self.metrics["active_threads"].append(active_threads)
        self.metrics["timestamp"].append(timestamp)

        # Keep only last 1000 data points to prevent memory bloat
        max_points = 1000
        for key in self.metrics:
            if len(self.metrics[key]) > max_points:
                self.metrics[key] = self.metrics[key][-max_points:]

    def log_operation(self, operation_name: str, start_time: float, end_time: float,
                     metadata: Optional[Dict[str, Any]] = None):
        """Log operation performance metrics."""
        duration = end_time - start_time

        operation_data = {
            "operation": operation_name,
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "metadata": metadata or {}
        }

        self.operation_metrics[operation_name].append(operation_data)

        # Keep only last 1000 operations per type
        if len(self.operation_metrics[operation_name]) > 1000:
            self.operation_metrics[operation_name] = self.operation_metrics[operation_name][-1000:]

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.metrics["timestamp"]:
            return {"error": "No metrics collected yet"}

        latest_idx = -1

        return {
            "timestamp": self.metrics["timestamp"][latest_idx],
            "cpu_percent": self.metrics["cpu_percent"][latest_idx],
            "memory_percent": self.metrics["memory_percent"][latest_idx],
            "memory_mb": self.metrics["memory_mb"][latest_idx],
            "disk_usage": self.metrics["disk_usage"][latest_idx],
            "network_connections": self.metrics["network_connections"][latest_idx],
            "active_threads": self.metrics["active_threads"][latest_idx]
        }

    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        if not self.metrics["timestamp"]:
            return {"error": "No metrics collected yet"}

        # Calculate how many data points to include (assuming 5-second intervals)
        points_per_hour = 3600 / self.log_interval
        points_to_include = int(hours * points_per_hour)

        # Get recent data
        recent_data = {}
        for key in self.metrics:
            recent_data[key] = self.metrics[key][-points_to_include:] if len(self.metrics[key]) >= points_to_include else self.metrics[key]

        if not recent_data["timestamp"]:
            return {"error": "Insufficient data for requested time period"}

        summary = {
            "time_period_hours": hours,
            "data_points": len(recent_data["timestamp"]),
            "cpu": {
                "avg_percent": sum(recent_data["cpu_percent"]) / len(recent_data["cpu_percent"]),
                "max_percent": max(recent_data["cpu_percent"]),
                "min_percent": min(recent_data["cpu_percent"])
            },
            "memory": {
                "avg_percent": sum(recent_data["memory_percent"]) / len(recent_data["memory_percent"]),
                "avg_mb": sum(recent_data["memory_mb"]) / len(recent_data["memory_mb"]),
                "max_mb": max(recent_data["memory_mb"]),
                "min_mb": min(recent_data["memory_mb"])
            },
            "disk": {
                "avg_usage_percent": sum(recent_data["disk_usage"]) / len(recent_data["disk_usage"])
            },
            "system": {
                "avg_threads": sum(recent_data["active_threads"]) / len(recent_data["active_threads"]),
                "avg_network_connections": sum(recent_data["network_connections"]) / len(recent_data["network_connections"])
            }
        }

        return summary

    def get_operation_performance(self, operation_name: Optional[str] = None,
                                hours: int = 1) -> Dict[str, Any]:
        """Get operation performance metrics."""
        cutoff_time = time.time() - (hours * 3600)

        if operation_name:
            # Specific operation
            operations = [op for op in self.operation_metrics.get(operation_name, [])
                         if op["start_time"] > cutoff_time]

            if not operations:
                return {"error": f"No {operation_name} operations found in the last {hours} hours"}

            durations = [op["duration"] for op in operations]

            return {
                "operation": operation_name,
                "time_period_hours": hours,
                "total_operations": len(operations),
                "avg_duration": sum(durations) / len(durations),
                "max_duration": max(durations),
                "min_duration": min(durations),
                "total_time_spent": sum(durations),
                "operations_per_second": len(operations) / (hours * 3600)
            }
        else:
            # All operations summary
            all_operations = []
            for op_name, ops in self.operation_metrics.items():
                recent_ops = [op for op in ops if op["start_time"] > cutoff_time]
                if recent_ops:
                    durations = [op["duration"] for op in recent_ops]
                    all_operations.append({
                        "operation": op_name,
                        "count": len(recent_ops),
                        "avg_duration": sum(durations) / len(durations),
                        "total_time": sum(durations)
                    })

            return {
                "time_period_hours": hours,
                "operations": all_operations,
                "total_operation_types": len(all_operations)
            }

    def save_performance_report(self, filename: Optional[str] = None) -> str:
        """Save comprehensive performance report."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"

        filepath = os.path.join(self.performance_logs, filename)

        report = {
            "generated_at": datetime.now().isoformat(),
            "system_summary": self.get_performance_summary(hours=1),
            "operation_performance": self.get_operation_performance(hours=1),
            "current_metrics": self.get_current_metrics(),
            "monitoring_status": "active" if self.is_monitoring else "inactive"
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Performance report saved to {filepath}")
        return filepath

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status based on performance metrics."""
        current = self.get_current_metrics()

        if "error" in current:
            return {"status": "unknown", "message": "No performance data available"}

        health_status = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }

        # CPU health check
        cpu_percent = current["cpu_percent"]
        if cpu_percent > 90:
            health_status["checks"]["cpu"] = {"status": "critical", "value": cpu_percent}
            health_status["status"] = "critical"
        elif cpu_percent > 75:
            health_status["checks"]["cpu"] = {"status": "warning", "value": cpu_percent}
            if health_status["status"] == "healthy":
                health_status["status"] = "warning"
        else:
            health_status["checks"]["cpu"] = {"status": "healthy", "value": cpu_percent}

        # Memory health check
        memory_percent = current["memory_percent"]
        if memory_percent > 95:
            health_status["checks"]["memory"] = {"status": "critical", "value": memory_percent}
            health_status["status"] = "critical"
        elif memory_percent > 85:
            health_status["checks"]["memory"] = {"status": "warning", "value": memory_percent}
            if health_status["status"] == "healthy":
                health_status["status"] = "warning"
        else:
            health_status["checks"]["memory"] = {"status": "healthy", "value": memory_percent}

        # Disk health check
        disk_percent = current["disk_usage"]
        if disk_percent > 95:
            health_status["checks"]["disk"] = {"status": "critical", "value": disk_percent}
            health_status["status"] = "critical"
        elif disk_percent > 85:
            health_status["checks"]["disk"] = {"status": "warning", "value": disk_percent}
            if health_status["status"] == "healthy":
                health_status["status"] = "warning"
        else:
            health_status["checks"]["disk"] = {"status": "healthy", "value": disk_percent}

        return health_status

class PerformanceTimer:
    """
    Context manager for timing operations.
    """

    def __init__(self, profiler: PerformanceProfiler, operation_name: str,
                 metadata: Optional[Dict[str, Any]] = None):
        self.profiler = profiler
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.profiler.log_operation(
            self.operation_name,
            self.start_time,
            end_time,
            self.metadata
        )


def performance_monitor(operation_name: str, profiler: Optional[PerformanceProfiler] = None):
    """
    Decorator to monitor performance of functions.

    Args:
        operation_name: Name of the operation for logging
        profiler: PerformanceProfiler instance (if None, creates a global one)

    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get or create profiler
            prof = profiler or _get_global_profiler()

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()

                # Log successful operation
                prof.log_operation(
                    operation_name,
                    start_time,
                    end_time,
                    {
                        "function": func.__name__,
                        "module": func.__module__,
                        "success": True
                    }
                )

                return result

            except Exception as e:
                end_time = time.time()

                # Log failed operation
                prof.log_operation(
                    operation_name,
                    start_time,
                    end_time,
                    {
                        "function": func.__name__,
                        "module": func.__module__,
                        "success": False,
                        "error": str(e)
                    }
                )

                raise e

        return wrapper
    return decorator


# Global profiler instance
_global_profiler = None

def _get_global_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler

def set_global_profiler(profiler: PerformanceProfiler):
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler
