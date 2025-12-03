from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import structlog
from enum import Enum

logger = structlog.get_logger()

class ReasoningConsistency(Enum):
    """Consistency levels for peer reasoning validation."""
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    SUSPICIOUS = "suspicious"
    INSUFFICIENT_DATA = "insufficient_data"

@dataclass
class ReasoningTrace:
    """Trace of an agent's reasoning process."""
    agent_id: str
    input_prompt: str
    reasoning_steps: List[str]
    final_output: str
    tool_calls: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class PeerGuard:
    """
    PeerGuard Mutual Reasoning Defense Mechanism.

    Based on research showing 96% true positive rates and <10% false positives
    across GPT-4o and Llama 3 models. Uses agent-to-agent reasoning to detect
    illogical behavior indicative of prompt-based poisoning.
    """

    def __init__(self, consistency_threshold: float = 0.7):
        """
        Initialize PeerGuard defense.

        Args:
            consistency_threshold: Minimum consistency score (0-1) for validation
        """
        self.consistency_threshold = consistency_threshold
        self.reasoning_history: Dict[str, List[ReasoningTrace]] = {}
        self.peer_validations: List[Dict[str, Any]] = []

    def record_reasoning(self, trace: ReasoningTrace) -> None:
        """
        Record an agent's reasoning trace for peer validation.

        Args:
            trace: Reasoning trace to record
        """
        if trace.agent_id not in self.reasoning_history:
            self.reasoning_history[trace.agent_id] = []

        self.reasoning_history[trace.agent_id].append(trace)
        logger.info("Reasoning recorded",
                   agent_id=trace.agent_id,
                   steps=len(trace.reasoning_steps))

    def validate_reasoning(self, trace: ReasoningTrace,
                          peer_agent_id: str) -> Dict[str, Any]:
        """
        Have a peer agent validate another agent's reasoning.

        Args:
            trace: Reasoning trace to validate
            peer_agent_id: ID of the validating peer agent

        Returns:
            Validation results including consistency score
        """
        inconsistencies = []

        # Check 1: Reasoning steps should logically flow
        logical_flow_score = self._check_logical_flow(trace.reasoning_steps)

        # Check 2: Output should match reasoning conclusion
        output_consistency_score = self._check_output_consistency(
            trace.reasoning_steps, trace.final_output
        )

        # Check 3: Tool calls should be appropriate for the task
        tool_appropriateness_score = self._check_tool_appropriateness(
            trace.input_prompt, trace.tool_calls
        )

        # Check 4: Compare with historical behavior
        historical_consistency_score = self._check_historical_consistency(trace)

        # Aggregate scores
        overall_score = (
            logical_flow_score * 0.3 +
            output_consistency_score * 0.3 +
            tool_appropriateness_score * 0.2 +
            historical_consistency_score * 0.2
        )

        # Determine consistency level
        if overall_score >= self.consistency_threshold:
            consistency = ReasoningConsistency.CONSISTENT
        elif overall_score >= 0.4:
            consistency = ReasoningConsistency.SUSPICIOUS
        else:
            consistency = ReasoningConsistency.INCONSISTENT

        # Collect specific inconsistencies
        if logical_flow_score < 0.5:
            inconsistencies.append({
                "type": "illogical_reasoning_flow",
                "score": logical_flow_score,
                "severity": 1.0 - logical_flow_score
            })

        if output_consistency_score < 0.5:
            inconsistencies.append({
                "type": "output_reasoning_mismatch",
                "score": output_consistency_score,
                "severity": 1.0 - output_consistency_score
            })

        if tool_appropriateness_score < 0.5:
            inconsistencies.append({
                "type": "inappropriate_tool_usage",
                "score": tool_appropriateness_score,
                "severity": 1.0 - tool_appropriateness_score
            })

        validation_result = {
            "validated_agent": trace.agent_id,
            "peer_validator": peer_agent_id,
            "consistency": consistency.value,
            "overall_score": overall_score,
            "component_scores": {
                "logical_flow": logical_flow_score,
                "output_consistency": output_consistency_score,
                "tool_appropriateness": tool_appropriateness_score,
                "historical_consistency": historical_consistency_score
            },
            "inconsistencies": inconsistencies,
            "is_suspicious": consistency in [
                ReasoningConsistency.SUSPICIOUS,
                ReasoningConsistency.INCONSISTENT
            ]
        }

        self.peer_validations.append(validation_result)

        if validation_result["is_suspicious"]:
            logger.warning("Suspicious reasoning detected by peer",
                         validated_agent=trace.agent_id,
                         peer=peer_agent_id,
                         score=overall_score,
                         inconsistencies=len(inconsistencies))

        return validation_result

    def _check_logical_flow(self, reasoning_steps: List[str]) -> float:
        """
        Check if reasoning steps follow logical progression.
        Simplified heuristic implementation.

        Returns:
            Score from 0.0 (illogical) to 1.0 (logical)
        """
        if not reasoning_steps:
            return 0.0

        # Heuristic checks
        score = 1.0

        # Check for contradictions (simplified)
        negation_words = ["not", "never", "opposite", "contrary", "however"]
        contradiction_count = 0

        for i in range(len(reasoning_steps) - 1):
            step = reasoning_steps[i].lower()
            next_step = reasoning_steps[i + 1].lower()

            # Simple contradiction detection
            if any(word in next_step for word in negation_words):
                # Check if reversing previous statement
                contradiction_count += 1

        # Penalize excessive contradictions
        if len(reasoning_steps) > 1:
            contradiction_ratio = contradiction_count / len(reasoning_steps)
            score -= contradiction_ratio * 0.5

        # Check for repetition (may indicate stuck reasoning)
        unique_steps = set(reasoning_steps)
        repetition_penalty = 1.0 - (len(unique_steps) / len(reasoning_steps))
        score -= repetition_penalty * 0.3

        return max(0.0, min(1.0, score))

    def _check_output_consistency(self, reasoning_steps: List[str],
                                  final_output: str) -> float:
        """
        Check if final output aligns with reasoning steps.

        Returns:
            Consistency score from 0.0 to 1.0
        """
        if not reasoning_steps or not final_output:
            return 0.5  # Neutral score

        # Simple keyword overlap heuristic
        # In production, use semantic similarity with embeddings
        final_output_lower = final_output.lower()
        last_step_lower = reasoning_steps[-1].lower() if reasoning_steps else ""

        # Extract key terms (simplified)
        output_words = set(final_output_lower.split())
        last_step_words = set(last_step_lower.split())

        if not last_step_words:
            return 0.5

        # Calculate overlap
        overlap = len(output_words.intersection(last_step_words))
        overlap_ratio = overlap / len(last_step_words)

        return min(1.0, overlap_ratio * 2)  # Scale up, cap at 1.0

    def _check_tool_appropriateness(self, input_prompt: str,
                                   tool_calls: List[Dict[str, Any]]) -> float:
        """
        Check if tool calls are appropriate for the input.

        Returns:
            Appropriateness score from 0.0 to 1.0
        """
        if not tool_calls:
            return 1.0  # No tools is fine

        # Heuristic: check for suspicious tool patterns
        score = 1.0

        # Check for excessive tool calls
        if len(tool_calls) > 10:
            score -= 0.3

        # Check for repeated identical tool calls
        unique_calls = len(set(str(tc) for tc in tool_calls))
        if len(tool_calls) > 0:
            repetition_ratio = 1.0 - (unique_calls / len(tool_calls))
            score -= repetition_ratio * 0.4

        # Check for suspicious tool combinations
        tool_names = [tc.get("tool", "") for tc in tool_calls]

        # Example: mixing file operations with network operations might be suspicious
        has_file_ops = any("file" in name.lower() for name in tool_names)
        has_network_ops = any(
            term in name.lower()
            for name in tool_names
            for term in ["http", "request", "fetch", "download"]
        )

        if has_file_ops and has_network_ops and len(tool_calls) > 5:
            score -= 0.2  # Potentially exfiltration attempt

        return max(0.0, min(1.0, score))

    def _check_historical_consistency(self, trace: ReasoningTrace) -> float:
        """
        Compare reasoning with agent's historical patterns.

        Returns:
            Consistency score from 0.0 to 1.0
        """
        agent_history = self.reasoning_history.get(trace.agent_id, [])

        if len(agent_history) < 3:
            return 0.8  # Neutral-positive for new agents

        # Compare with historical averages
        historical_step_counts = [len(t.reasoning_steps) for t in agent_history]
        historical_tool_counts = [len(t.tool_calls) for t in agent_history]

        avg_steps = sum(historical_step_counts) / len(historical_step_counts)
        avg_tools = sum(historical_tool_counts) / len(historical_tool_counts)

        # Calculate deviations
        step_deviation = abs(len(trace.reasoning_steps) - avg_steps) / (avg_steps + 1)
        tool_deviation = abs(len(trace.tool_calls) - avg_tools) / (avg_tools + 1)

        # Score based on deviations (lower deviation = higher score)
        score = 1.0 - min(1.0, (step_deviation + tool_deviation) / 4)

        return score

    def cross_validate_agents(self, traces: List[ReasoningTrace]) -> Dict[str, Any]:
        """
        Perform cross-validation among multiple agents.

        Args:
            traces: List of reasoning traces from different agents

        Returns:
            Cross-validation results
        """
        if len(traces) < 2:
            return {
                "status": "insufficient_agents",
                "validated": False
            }

        # Each agent validates others
        validation_matrix = []

        for i, trace in enumerate(traces):
            for j, peer_trace in enumerate(traces):
                if i != j:
                    # Agent j validates agent i's reasoning
                    validation = self.validate_reasoning(trace, peer_trace.agent_id)
                    validation_matrix.append(validation)

        # Aggregate results
        suspicious_count = sum(1 for v in validation_matrix if v["is_suspicious"])
        total_validations = len(validation_matrix)

        suspicion_ratio = suspicious_count / total_validations if total_validations > 0 else 0

        # Flag if majority find reasoning suspicious
        is_compromised = suspicion_ratio > 0.5

        result = {
            "status": "complete",
            "validated": not is_compromised,
            "total_validations": total_validations,
            "suspicious_validations": suspicious_count,
            "suspicion_ratio": suspicion_ratio,
            "is_compromised": is_compromised,
            "individual_validations": validation_matrix
        }

        if is_compromised:
            logger.error("Cross-validation detected compromised agent(s)",
                        suspicion_ratio=suspicion_ratio,
                        agents=[t.agent_id for t in traces])

        return result

    def get_agent_reputation(self, agent_id: str,
                           recent_window: int = 20) -> float:
        """
        Calculate agent reputation based on peer validation history.

        Args:
            agent_id: Agent to evaluate
            recent_window: Number of recent validations to consider

        Returns:
            Reputation score from 0.0 (untrusted) to 1.0 (trusted)
        """
        relevant_validations = [
            v for v in self.peer_validations[-recent_window:]
            if v["validated_agent"] == agent_id
        ]

        if not relevant_validations:
            return 0.5  # Neutral for unknown agents

        # Average consistency scores
        avg_score = sum(v["overall_score"] for v in relevant_validations) / len(relevant_validations)

        return avg_score
