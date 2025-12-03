from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
from datetime import datetime, timedelta
import numpy as np
import structlog

logger = structlog.get_logger()

class BehavioralMonitor:
    """
    Monitors agent behavior for anomalies that may indicate prompt injection.
    Based on research showing 43.4% generalization gap in single-model detectors.
    Implements model-aware behavioral baselines.
    """

    def __init__(self, window_size: int = 100, anomaly_threshold: float = 2.5):
        """
        Initialize behavioral monitor.

        Args:
            window_size: Number of recent interactions to track
            anomaly_threshold: Standard deviations from mean to flag anomaly
        """
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold

        # Track per-agent metrics
        self.agent_metrics: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                "output_lengths": deque(maxlen=window_size),
                "response_times": deque(maxlen=window_size),
                "tool_calls": deque(maxlen=window_size),
                "message_counts": deque(maxlen=window_size),
                "sentiment_scores": deque(maxlen=window_size),
                "injection_scores": deque(maxlen=window_size),
            }
        )

        # Track inter-agent communication patterns
        self.communication_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Baseline profiles (learned from initial observations)
        self.baselines: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Anomaly history
        self.anomalies: deque = deque(maxlen=1000)

    def record_interaction(self, agent_id: str,
                          output_length: int,
                          response_time: float,
                          tool_calls: int = 0,
                          sentiment_score: Optional[float] = None,
                          injection_score: Optional[float] = None,
                          target_agent: Optional[str] = None) -> None:
        """
        Record a single agent interaction for baseline learning.

        Args:
            agent_id: Identifier for the agent
            output_length: Length of output text
            response_time: Time taken to generate response (seconds)
            tool_calls: Number of tool invocations
            sentiment_score: Optional sentiment score (-1 to 1)
            injection_score: Optional injection probability (0 to 1)
            target_agent: Optional destination agent for communication tracking
        """
        metrics = self.agent_metrics[agent_id]
        metrics["output_lengths"].append(output_length)
        metrics["response_times"].append(response_time)
        metrics["tool_calls"].append(tool_calls)

        if sentiment_score is not None:
            metrics["sentiment_scores"].append(sentiment_score)

        if injection_score is not None:
            metrics["injection_scores"].append(injection_score)

        # Track communication patterns
        if target_agent:
            self.communication_graph[agent_id][target_agent] += 1

        # Update baseline if we have enough data
        if len(metrics["output_lengths"]) >= 30 and agent_id not in self.baselines:
            self._compute_baseline(agent_id)

    def _compute_baseline(self, agent_id: str) -> None:
        """Compute baseline statistics for an agent."""
        metrics = self.agent_metrics[agent_id]
        baseline = {}

        for metric_name, values in metrics.items():
            if len(values) > 0:
                arr = np.array(list(values))
                baseline[metric_name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "median": float(np.median(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr))
                }

        self.baselines[agent_id] = baseline
        logger.info("Baseline computed", agent_id=agent_id, metrics=list(baseline.keys()))

    def detect_anomaly(self, agent_id: str,
                      output_length: int,
                      response_time: float,
                      tool_calls: int = 0,
                      sentiment_score: Optional[float] = None,
                      injection_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect behavioral anomalies in agent interaction.

        Returns:
            Dictionary with anomaly detection results
        """
        # First record the interaction
        self.record_interaction(
            agent_id, output_length, response_time, tool_calls,
            sentiment_score, injection_score
        )

        # If no baseline yet, can't detect anomalies
        if agent_id not in self.baselines:
            return {
                "is_anomalous": False,
                "reason": "insufficient_baseline_data",
                "anomalies": []
            }

        baseline = self.baselines[agent_id]
        anomalies = []

        # Check each metric for anomalies
        current_values = {
            "output_lengths": output_length,
            "response_times": response_time,
            "tool_calls": tool_calls,
        }

        if sentiment_score is not None:
            current_values["sentiment_scores"] = sentiment_score

        if injection_score is not None:
            current_values["injection_scores"] = injection_score

        for metric_name, current_value in current_values.items():
            if metric_name not in baseline:
                continue

            mean = baseline[metric_name]["mean"]
            std = baseline[metric_name]["std"]

            # Skip if std is too small (constant behavior)
            if std < 1e-6:
                continue

            # Calculate z-score
            z_score = abs((current_value - mean) / std)

            if z_score > self.anomaly_threshold:
                anomalies.append({
                    "metric": metric_name,
                    "current_value": current_value,
                    "expected_mean": mean,
                    "z_score": z_score,
                    "severity": min(z_score / 5.0, 1.0)  # Normalize to 0-1
                })

        is_anomalous = len(anomalies) > 0

        if is_anomalous:
            anomaly_event = {
                "timestamp": datetime.utcnow(),
                "agent_id": agent_id,
                "anomalies": anomalies
            }
            self.anomalies.append(anomaly_event)
            logger.warning("Behavioral anomaly detected",
                         agent_id=agent_id,
                         num_anomalies=len(anomalies))

        return {
            "is_anomalous": is_anomalous,
            "anomalies": anomalies,
            "total_severity": sum(a["severity"] for a in anomalies),
            "agent_id": agent_id
        }

    def detect_communication_anomaly(self, source_agent: str,
                                    target_agent: str) -> Dict[str, Any]:
        """
        Detect anomalous communication patterns between agents.

        Returns:
            Dictionary indicating if communication is unexpected
        """
        # Check if this is a new communication path
        total_communications = sum(self.communication_graph[source_agent].values())

        if total_communications < 10:
            return {
                "is_anomalous": False,
                "reason": "insufficient_communication_history"
            }

        # Check if this destination is unusual
        target_count = self.communication_graph[source_agent][target_agent]
        expected_frequency = target_count / total_communications

        # Flag if this is a very rare or new communication path
        is_anomalous = expected_frequency < 0.05 and target_count < 3

        if is_anomalous:
            logger.warning("Unusual communication pattern",
                         source=source_agent,
                         target=target_agent,
                         frequency=expected_frequency)

        return {
            "is_anomalous": is_anomalous,
            "frequency": expected_frequency,
            "total_count": target_count,
            "source_agent": source_agent,
            "target_agent": target_agent
        }

    def detect_output_shift(self, agent_id: str, current_outputs: List[str],
                           sample_size: int = 20) -> Dict[str, Any]:
        """
        Detect sudden shifts in output distribution using recent samples.

        Args:
            agent_id: Agent identifier
            current_outputs: Recent output strings
            sample_size: Number of samples to analyze

        Returns:
            Distribution shift detection results
        """
        if agent_id not in self.baselines:
            return {"is_shifted": False, "reason": "no_baseline"}

        if len(current_outputs) < sample_size:
            return {"is_shifted": False, "reason": "insufficient_samples"}

        # Simple distribution shift: compare recent length distribution
        recent_lengths = [len(out) for out in current_outputs[-sample_size:]]
        baseline_lengths = list(self.agent_metrics[agent_id]["output_lengths"])

        if len(baseline_lengths) < sample_size:
            return {"is_shifted": False, "reason": "insufficient_baseline"}

        # Use Kolmogorov-Smirnov test statistic (simplified)
        recent_mean = np.mean(recent_lengths)
        baseline_mean = np.mean(baseline_lengths)
        recent_std = np.std(recent_lengths)
        baseline_std = np.std(baseline_lengths)

        # Check for significant shift
        mean_shift = abs(recent_mean - baseline_mean) / (baseline_std + 1e-6)
        std_shift = abs(recent_std - baseline_std) / (baseline_std + 1e-6)

        is_shifted = mean_shift > 2.0 or std_shift > 1.5

        if is_shifted:
            logger.warning("Output distribution shift detected",
                         agent_id=agent_id,
                         mean_shift=mean_shift,
                         std_shift=std_shift)

        return {
            "is_shifted": is_shifted,
            "mean_shift": mean_shift,
            "std_shift": std_shift,
            "recent_mean": recent_mean,
            "baseline_mean": baseline_mean
        }

    def get_agent_risk_score(self, agent_id: str,
                            time_window: timedelta = timedelta(minutes=10)) -> float:
        """
        Calculate overall risk score for an agent based on recent anomalies.

        Args:
            agent_id: Agent identifier
            time_window: Time window for recent anomalies

        Returns:
            Risk score from 0.0 (safe) to 1.0 (high risk)
        """
        cutoff_time = datetime.utcnow() - time_window

        # Count recent anomalies for this agent
        recent_anomalies = [
            a for a in self.anomalies
            if a["agent_id"] == agent_id and a["timestamp"] > cutoff_time
        ]

        if not recent_anomalies:
            return 0.0

        # Calculate risk based on frequency and severity
        total_severity = sum(
            sum(anomaly["severity"] for anomaly in event["anomalies"])
            for event in recent_anomalies
        )

        # Normalize by number of events and time
        risk_score = min(total_severity / 10.0, 1.0)

        return risk_score

    def get_anomaly_report(self, agent_id: Optional[str] = None,
                          limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent anomaly events.

        Args:
            agent_id: Optional filter by agent
            limit: Maximum number of events to return

        Returns:
            List of anomaly events
        """
        events = list(self.anomalies)

        if agent_id:
            events = [e for e in events if e["agent_id"] == agent_id]

        return events[-limit:]
