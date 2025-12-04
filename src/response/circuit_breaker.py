import time
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import structlog

logger = structlog.get_logger()

class AlertSeverity(Enum):
    """Tiered alert severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Individual alert with metadata for correlation."""
    timestamp: float
    severity: AlertSeverity
    source: str  # Which detector raised this
    category: str  # e.g., "pattern_match", "embedding_score", "behavioral"
    details: Dict[str, Any] = field(default_factory=dict)
    agent_id: Optional[str] = None
    correlated_alerts: List[str] = field(default_factory=list)

class CircuitBreaker:
    """
    Circuit breaker pattern to stop automated attacks.
    Tracks failure rates and opens the circuit (blocking all traffic) if threshold is exceeded.
    """
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60, time_window: int = 60,
                 critical_threshold: int = 2, correlation_window: int = 30, threshold: int = None):
        """
        Initialize the safety circuit.
        
        Args:
            failure_threshold: Number of failures allowed before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery (half-open)
            time_window: Sliding window in seconds for counting failures
            critical_threshold: Number of critical alerts before immediate trip
            correlation_window: Time window for correlating alerts
            threshold: Alias for failure_threshold (for backwards compatibility)
        """
        # Support 'threshold' as alias for backwards compatibility
        if threshold is not None:
            failure_threshold = threshold
            
        self.failure_threshold = failure_threshold
        self.threshold = failure_threshold  # Alias for legacy code
        self.recovery_timeout = recovery_timeout
        self.time_window = time_window
        self.critical_threshold = critical_threshold
        self.correlation_window = correlation_window

        self.events: List[float] = []  # Legacy simple events
        self.alerts: List[Alert] = []  # Enhanced alerts
        self._is_open = False

        # Alert correlation state
        self.correlated_groups: List[List[Alert]] = []

    def record_event(self):
        """Record a suspicious event (legacy method)."""
        current_time = time.time()
        self.events.append(current_time)
        self._cleanup(current_time)

        if len(self.events) >= self.threshold:
            if not self._is_open:
                logger.critical("Circuit breaker tripped!",
                              threshold=self.threshold,
                              window=self.time_window)
                self._is_open = True

    def record_alert(self, severity: AlertSeverity, source: str, category: str,
                    details: Optional[Dict[str, Any]] = None,
                    agent_id: Optional[str] = None) -> Alert:
        """
        Record a tiered alert with correlation support.

        Args:
            severity: Alert severity level
            source: Which component raised this alert
            category: Alert category for correlation
            details: Additional context
            agent_id: Associated agent if applicable

        Returns:
            Created alert object
        """
        current_time = time.time()

        alert = Alert(
            timestamp=current_time,
            severity=severity,
            source=source,
            category=category,
            details=details or {},
            agent_id=agent_id
        )

        self.alerts.append(alert)
        self._cleanup_alerts(current_time)

        # Correlate with recent alerts
        self._correlate_alert(alert)

        # Log based on severity
        log_data = {
            "source": source,
            "category": category,
            "agent_id": agent_id,
            "details": details
        }

        if severity == AlertSeverity.INFO:
            logger.info("Alert recorded", **log_data)
        elif severity == AlertSeverity.LOW:
            logger.info("Low severity alert", **log_data)
        elif severity == AlertSeverity.MEDIUM:
            logger.warning("Medium severity alert", **log_data)
        elif severity == AlertSeverity.HIGH:
            logger.warning("High severity alert", **log_data)
        elif severity == AlertSeverity.CRITICAL:
            logger.error("Critical alert", **log_data)

        # Check if circuit should trip
        self._check_trip_conditions()

        return alert

    def _correlate_alert(self, new_alert: Alert) -> None:
        """
        Correlate new alert with recent alerts to identify patterns.

        Groups alerts that:
        - Occur within correlation window
        - Share the same category or agent
        - Indicate coordinated attack
        """
        cutoff_time = new_alert.timestamp - self.correlation_window
        recent_alerts = [
            a for a in self.alerts
            if a.timestamp > cutoff_time and a != new_alert
        ]

        # Find correlated alerts
        correlated = []
        for alert in recent_alerts:
            # Correlation criteria
            same_category = alert.category == new_alert.category
            same_agent = alert.agent_id and alert.agent_id == new_alert.agent_id
            high_severity = alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]

            if same_category or same_agent or high_severity:
                correlated.append(alert)

        # Create correlation group if multiple related alerts
        if len(correlated) >= 2:
            group = correlated + [new_alert]
            self.correlated_groups.append(group)

            # Update alert with correlation info
            new_alert.correlated_alerts = [a.source for a in correlated]

            logger.warning("Correlated alert group detected",
                         group_size=len(group),
                         categories=list(set(a.category for a in group)),
                         time_span=new_alert.timestamp - min(a.timestamp for a in group))

    def _check_trip_conditions(self) -> None:
        """Check if circuit breaker should trip based on alert analysis."""
        if self._is_open:
            return

        current_time = time.time()
        cutoff_time = current_time - self.time_window

        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]

        # Count by severity
        critical_count = sum(1 for a in recent_alerts if a.severity == AlertSeverity.CRITICAL)
        high_count = sum(1 for a in recent_alerts if a.severity == AlertSeverity.HIGH)

        # Immediate trip conditions
        if critical_count >= self.critical_threshold:
            logger.critical("Circuit breaker tripped due to critical alerts!",
                          critical_count=critical_count,
                          threshold=self.critical_threshold)
            self._is_open = True
            return

        # Trip on high volume of high-severity alerts
        if high_count + critical_count * 2 >= self.threshold:
            logger.critical("Circuit breaker tripped due to high-severity alert volume!",
                          high_count=high_count,
                          critical_count=critical_count)
            self._is_open = True
            return

        # Trip on total alert volume
        if len(recent_alerts) >= self.threshold * 1.5:
            logger.critical("Circuit breaker tripped due to alert volume!",
                          alert_count=len(recent_alerts),
                          threshold=self.threshold)
            self._is_open = True

    def is_open(self) -> bool:
        """
        Check if the circuit breaker is open (tripped).

        Returns:
            True if open (should stop processing), False otherwise.
        """
        current_time = time.time()
        self._cleanup(current_time)
        self._cleanup_alerts(current_time)
        return self._is_open

    def reset(self):
        """Reset the circuit breaker manually."""
        self.events = []
        self.alerts = []
        self.correlated_groups = []
        self._is_open = False
        logger.info("Circuit breaker reset")

    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent alerts grouped by severity and category.

        Returns:
            Alert summary statistics
        """
        current_time = time.time()
        cutoff_time = current_time - self.time_window
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]

        # Group by severity
        by_severity = {
            severity: sum(1 for a in recent_alerts if a.severity == severity)
            for severity in AlertSeverity
        }

        # Group by category
        categories = {}
        for alert in recent_alerts:
            categories[alert.category] = categories.get(alert.category, 0) + 1

        # Group by agent
        agents = {}
        for alert in recent_alerts:
            if alert.agent_id:
                agents[alert.agent_id] = agents.get(alert.agent_id, 0) + 1

        return {
            "total_alerts": len(recent_alerts),
            "by_severity": {s.value: count for s, count in by_severity.items()},
            "by_category": categories,
            "by_agent": agents,
            "correlated_groups": len(self.correlated_groups),
            "circuit_status": "open" if self._is_open else "closed"
        }

    def get_correlated_alerts(self) -> List[List[Dict[str, Any]]]:
        """
        Get all correlated alert groups.

        Returns:
            List of alert groups with correlation details
        """
        return [
            [
                {
                    "timestamp": datetime.fromtimestamp(a.timestamp).isoformat(),
                    "severity": a.severity.value,
                    "source": a.source,
                    "category": a.category,
                    "agent_id": a.agent_id
                }
                for a in group
            ]
            for group in self.correlated_groups
        ]

    def _cleanup(self, current_time: float):
        """Remove events outside the time window."""
        cutoff = current_time - self.time_window
        self.events = [t for t in self.events if t > cutoff]

        # Auto-recovery: close circuit if no recent events
        if self._is_open and len(self.events) == 0 and len(self.alerts) == 0:
            logger.info("Circuit breaker auto-recovering (no recent events)")
            self._is_open = False

    def _cleanup_alerts(self, current_time: float):
        """Remove alerts outside the time window."""
        cutoff = current_time - self.time_window
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff]

        self.correlated_groups = [
            group for group in self.correlated_groups
            if any(a.timestamp > cutoff for a in group)
        ]

    def get_recovery_time(self) -> float:
        """
        Get the remaining time in seconds until the circuit breaker recovers.
        
        Returns:
            Remaining seconds, or 0 if not open or ready to recover.
        """
        if not self._is_open:
            return 0.0
            
        current_time = time.time()
        
        # Find the latest event or alert timestamp
        last_event_time = max(self.events) if self.events else 0
        last_alert_time = max((a.timestamp for a in self.alerts), default=0)
        
        last_activity = max(last_event_time, last_alert_time)
        
        if last_activity == 0:
            return 0.0
            
        recovery_time = last_activity + self.time_window
        remaining = max(0.0, recovery_time - current_time)
        
        return remaining
