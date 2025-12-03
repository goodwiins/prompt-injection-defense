from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()

class KPIEvaluator:
    """
    KPI (Key Performance Indicator) Evaluator for real-time security monitoring.

    Tracks and evaluates:
    - Detection accuracy and latency
    - False positive/negative rates
    - System throughput
    - Agent health metrics
    - Compliance scores
    """

    def __init__(self, evaluation_window: int = 3600):
        """
        Initialize KPI evaluator.

        Args:
            evaluation_window: Time window in seconds for metrics calculation
        """
        self.evaluation_window = evaluation_window

        # Detection metrics
        self.detections: List[Dict[str, Any]] = []

        # Performance metrics
        self.latencies: List[float] = []

        # Agent health
        self.agent_health: Dict[str, Dict[str, Any]] = {}

        # Compliance tracking
        self.compliance_checks: List[Dict[str, Any]] = []

    def record_detection(self, is_true_positive: bool, is_false_positive: bool,
                        is_false_negative: bool, latency_ms: float,
                        agent_id: Optional[str] = None) -> None:
        """
        Record a detection event.

        Args:
            is_true_positive: Correctly identified attack
            is_false_positive: Incorrectly flagged benign input
            is_false_negative: Missed an actual attack
            latency_ms: Detection latency in milliseconds
            agent_id: Agent that performed detection
        """
        self.detections.append({
            "timestamp": datetime.utcnow(),
            "is_true_positive": is_true_positive,
            "is_false_positive": is_false_positive,
            "is_false_negative": is_false_negative,
            "latency_ms": latency_ms,
            "agent_id": agent_id
        })

        self.latencies.append(latency_ms)

        # Cleanup old entries
        self._cleanup()

    def record_agent_health(self, agent_id: str, status: str,
                           cpu_usage: Optional[float] = None,
                           memory_usage: Optional[float] = None,
                           error_count: int = 0) -> None:
        """
        Record agent health metrics.

        Args:
            agent_id: Agent identifier
            status: Agent status (healthy, degraded, failed)
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            error_count: Number of errors in current period
        """
        self.agent_health[agent_id] = {
            "timestamp": datetime.utcnow(),
            "status": status,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "error_count": error_count
        }

    def record_compliance(self, policy_name: str, compliant: bool,
                         severity: str, agent_id: Optional[str] = None) -> None:
        """
        Record compliance check result.

        Args:
            policy_name: Name of policy checked
            compliant: Whether check passed
            severity: Severity level (low, medium, high, critical)
            agent_id: Agent being evaluated
        """
        self.compliance_checks.append({
            "timestamp": datetime.utcnow(),
            "policy_name": policy_name,
            "compliant": compliant,
            "severity": severity,
            "agent_id": agent_id
        })

        self._cleanup()

    def get_detection_kpis(self) -> Dict[str, Any]:
        """
        Calculate detection-related KPIs.

        Returns:
            Dictionary with detection metrics
        """
        if not self.detections:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0
            }

        recent = self._get_recent_detections()

        tp = sum(1 for d in recent if d["is_true_positive"])
        fp = sum(1 for d in recent if d["is_false_positive"])
        fn = sum(1 for d in recent if d["is_false_negative"])
        tn = len(recent) - tp - fp - fn  # True negatives

        total = len(recent)

        # Calculate metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Latency metrics
        recent_latencies = [d["latency_ms"] for d in recent]
        avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0.0

        sorted_latencies = sorted(recent_latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0.0
        p99_latency = sorted_latencies[p99_idx] if sorted_latencies else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "total_detections": total
        }

    def get_agent_health_kpis(self) -> Dict[str, Any]:
        """
        Calculate agent health KPIs.

        Returns:
            Dictionary with agent health metrics
        """
        if not self.agent_health:
            return {
                "total_agents": 0,
                "healthy_agents": 0,
                "degraded_agents": 0,
                "failed_agents": 0,
                "avg_cpu_usage": 0.0,
                "avg_memory_usage": 0.0,
                "total_errors": 0
            }

        total = len(self.agent_health)
        healthy = sum(1 for h in self.agent_health.values() if h["status"] == "healthy")
        degraded = sum(1 for h in self.agent_health.values() if h["status"] == "degraded")
        failed = sum(1 for h in self.agent_health.values() if h["status"] == "failed")

        cpu_values = [h["cpu_usage"] for h in self.agent_health.values() if h["cpu_usage"] is not None]
        mem_values = [h["memory_usage"] for h in self.agent_health.values() if h["memory_usage"] is not None]

        avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
        avg_mem = sum(mem_values) / len(mem_values) if mem_values else 0.0
        total_errors = sum(h["error_count"] for h in self.agent_health.values())

        return {
            "total_agents": total,
            "healthy_agents": healthy,
            "degraded_agents": degraded,
            "failed_agents": failed,
            "health_rate": healthy / total if total > 0 else 0.0,
            "avg_cpu_usage": avg_cpu,
            "avg_memory_usage": avg_mem,
            "total_errors": total_errors
        }

    def get_compliance_kpis(self) -> Dict[str, Any]:
        """
        Calculate compliance KPIs.

        Returns:
            Dictionary with compliance metrics
        """
        if not self.compliance_checks:
            return {
                "total_checks": 0,
                "compliant_checks": 0,
                "compliance_rate": 0.0,
                "violations_by_severity": {}
            }

        recent = self._get_recent_compliance()

        total = len(recent)
        compliant = sum(1 for c in recent if c["compliant"])

        violations_by_severity = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0
        }

        for check in recent:
            if not check["compliant"]:
                severity = check["severity"]
                violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1

        return {
            "total_checks": total,
            "compliant_checks": compliant,
            "compliance_rate": compliant / total if total > 0 else 0.0,
            "violations_by_severity": violations_by_severity,
            "non_compliant_checks": total - compliant
        }

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive KPI report.

        Returns:
            Complete KPI dashboard data
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "evaluation_window_seconds": self.evaluation_window,
            "detection_kpis": self.get_detection_kpis(),
            "agent_health_kpis": self.get_agent_health_kpis(),
            "compliance_kpis": self.get_compliance_kpis()
        }

    def _get_recent_detections(self) -> List[Dict[str, Any]]:
        """Get detections within evaluation window."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.evaluation_window)
        return [d for d in self.detections if d["timestamp"] > cutoff]

    def _get_recent_compliance(self) -> List[Dict[str, Any]]:
        """Get compliance checks within evaluation window."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.evaluation_window)
        return [c for c in self.compliance_checks if c["timestamp"] > cutoff]

    def _cleanup(self) -> None:
        """Remove old entries outside evaluation window."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.evaluation_window)

        self.detections = [d for d in self.detections if d["timestamp"] > cutoff]
        self.compliance_checks = [c for c in self.compliance_checks if c["timestamp"] > cutoff]

        # Keep only recent latencies
        recent_detections = self._get_recent_detections()
        self.latencies = [d["latency_ms"] for d in recent_detections]

    def reset(self) -> None:
        """Reset all metrics."""
        self.detections = []
        self.latencies = []
        self.agent_health = {}
        self.compliance_checks = []
        logger.info("KPI evaluator reset")
