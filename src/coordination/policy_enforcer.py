from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

class PolicyAction(Enum):
    """Policy enforcement actions."""
    ALLOW = "allow"
    SANITIZE = "sanitize"
    FLAG = "flag"
    BLOCK = "block"
    QUARANTINE = "quarantine"

@dataclass
class Policy:
    """Security policy definition."""
    name: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    conditions: List[str]  # List of condition types
    action: PolicyAction
    metadata: Dict[str, Any]

class PolicyEnforcer:
    """
    Policy Enforcer agent for compliance verification.

    Validates that messages and agent interactions comply with
    defined security policies. Implements metadata tagging and
    compliance tracking as recommended in multi-agent defense research.
    """

    def __init__(self):
        """Initialize policy enforcer with default policies."""
        self.policies: Dict[str, Policy] = {}
        self.violations: List[Dict[str, Any]] = []
        self._initialize_default_policies()

    def _initialize_default_policies(self) -> None:
        """Set up default security policies."""

        # Policy 1: High injection score
        self.add_policy(Policy(
            name="high_injection_score",
            description="Block messages with high injection probability",
            severity="critical",
            conditions=["injection_score > 0.85"],
            action=PolicyAction.BLOCK,
            metadata={"max_injection_score": 0.85}
        ))

        # Policy 2: Low trust agent
        self.add_policy(Policy(
            name="low_trust_agent",
            description="Flag messages from low-trust agents",
            severity="high",
            conditions=["trust_level < 0.3"],
            action=PolicyAction.FLAG,
            metadata={"min_trust_level": 0.3}
        ))

        # Policy 3: Excessive message hops
        self.add_policy(Policy(
            name="excessive_hops",
            description="Quarantine messages with too many hops",
            severity="medium",
            conditions=["hop_count > 10"],
            action=PolicyAction.QUARANTINE,
            metadata={"max_hops": 10}
        ))

        # Policy 4: Signature verification failure
        self.add_policy(Policy(
            name="invalid_signature",
            description="Block messages with invalid signatures",
            severity="critical",
            conditions=["signature_valid == False"],
            action=PolicyAction.BLOCK,
            metadata={}
        ))

        # Policy 5: Suspicious pattern matches
        self.add_policy(Policy(
            name="pattern_injection",
            description="Sanitize messages matching injection patterns",
            severity="high",
            conditions=["matched_patterns.length > 0"],
            action=PolicyAction.SANITIZE,
            metadata={}
        ))

        # Policy 6: Behavioral anomaly
        self.add_policy(Policy(
            name="behavioral_anomaly",
            description="Flag agents exhibiting anomalous behavior",
            severity="medium",
            conditions=["behavioral_anomaly == True"],
            action=PolicyAction.FLAG,
            metadata={}
        ))

        # Policy 7: Rapid message rate
        self.add_policy(Policy(
            name="rate_limit",
            description="Rate limit agents sending too many messages",
            severity="medium",
            conditions=["message_rate > 100/minute"],
            action=PolicyAction.SANITIZE,
            metadata={"max_rate": 100}
        ))

        logger.info("Default policies initialized", policy_count=len(self.policies))

    def add_policy(self, policy: Policy) -> None:
        """
        Add or update a security policy.

        Args:
            policy: Policy to add
        """
        self.policies[policy.name] = policy
        logger.info("Policy added", policy_name=policy.name, action=policy.action.value)

    def remove_policy(self, policy_name: str) -> bool:
        """
        Remove a security policy.

        Args:
            policy_name: Name of policy to remove

        Returns:
            True if policy was removed
        """
        if policy_name in self.policies:
            del self.policies[policy_name]
            logger.info("Policy removed", policy_name=policy_name)
            return True
        return False

    def enforce(self, message_data: Dict[str, Any],
                agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enforce all policies on a message.

        Args:
            message_data: Message data to evaluate
            agent_id: Associated agent ID

        Returns:
            Enforcement result with actions to take
        """
        violations = []
        recommended_action = PolicyAction.ALLOW
        max_severity_score = 0

        severity_map = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }

        for policy_name, policy in self.policies.items():
            # Evaluate policy conditions
            if self._evaluate_policy(policy, message_data):
                violations.append({
                    "policy": policy_name,
                    "severity": policy.severity,
                    "action": policy.action.value,
                    "description": policy.description
                })

                # Update recommended action to most severe
                severity_score = severity_map.get(policy.severity, 0)
                if severity_score > max_severity_score:
                    max_severity_score = severity_score
                    recommended_action = policy.action

        # Record violations
        if violations:
            violation_record = {
                "agent_id": agent_id,
                "violations": violations,
                "action_taken": recommended_action.value,
                "message_summary": {
                    k: v for k, v in message_data.items()
                    if k in ["message_id", "source_agent", "destination_agent"]
                }
            }
            self.violations.append(violation_record)

            logger.warning("Policy violations detected",
                         agent_id=agent_id,
                         violation_count=len(violations),
                         action=recommended_action.value)

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "recommended_action": recommended_action.value,
            "severity_level": max_severity_score,
            "requires_intervention": max_severity_score >= 3
        }

    def _evaluate_policy(self, policy: Policy, data: Dict[str, Any]) -> bool:
        """
        Evaluate if data violates a policy.

        Args:
            policy: Policy to evaluate
            data: Message data

        Returns:
            True if policy is violated
        """
        # Simple condition evaluation
        # In production, use a proper policy engine

        for condition in policy.conditions:
            # Parse condition (simplified)
            if "injection_score >" in condition:
                threshold = float(condition.split(">")[1].strip())
                injection_score = data.get("injection_score", 0.0)
                if injection_score > threshold:
                    return True

            elif "trust_level <" in condition:
                threshold = float(condition.split("<")[1].strip())
                trust_level = data.get("trust_level", 1.0)
                if trust_level < threshold:
                    return True

            elif "hop_count >" in condition:
                threshold = int(condition.split(">")[1].strip())
                hop_count = data.get("hop_count", 0)
                if hop_count > threshold:
                    return True

            elif "signature_valid ==" in condition:
                if "signature_valid" in data:
                    expected = "True" in condition
                    signature_valid = data.get("signature_valid")
                    if signature_valid != expected:
                        return True
                else: # Don't trigger if signature is not present
                    return False

            elif "matched_patterns.length >" in condition:
                threshold = int(condition.split(">")[1].strip())
                matched_patterns = data.get("matched_patterns", [])
                if len(matched_patterns) > threshold:
                    return True

            elif "behavioral_anomaly ==" in condition:
                expected = "True" in condition
                behavioral_anomaly = data.get("behavioral_anomaly", False)
                if behavioral_anomaly == expected:
                    return True

            elif "message_rate >" in condition:
                # Extract rate limit (e.g., "100/minute")
                threshold_str = condition.split(">")[1].strip()
                threshold = int(threshold_str.split("/")[0])
                message_rate = data.get("message_rate", 0)
                if message_rate > threshold:
                    return True

        return False

    def get_compliance_report(self, agent_id: Optional[str] = None,
                             limit: int = 100) -> Dict[str, Any]:
        """
        Generate compliance report.

        Args:
            agent_id: Optional filter by agent
            limit: Maximum violations to include

        Returns:
            Compliance statistics and recent violations
        """
        filtered_violations = self.violations[-limit:]

        if agent_id:
            filtered_violations = [
                v for v in filtered_violations
                if v.get("agent_id") == agent_id
            ]

        # Count by severity
        by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for violation in filtered_violations:
            for v in violation["violations"]:
                severity = v["severity"]
                by_severity[severity] = by_severity.get(severity, 0) + 1

        # Count by policy
        by_policy = {}
        for violation in filtered_violations:
            for v in violation["violations"]:
                policy = v["policy"]
                by_policy[policy] = by_policy.get(policy, 0) + 1

        # Count by action
        by_action = {}
        for violation in filtered_violations:
            action = violation["action_taken"]
            by_action[action] = by_action.get(action, 0) + 1

        return {
            "total_violations": len(filtered_violations),
            "by_severity": by_severity,
            "by_policy": by_policy,
            "by_action": by_action,
            "recent_violations": filtered_violations[-10:],  # Last 10
            "compliance_rate": self._calculate_compliance_rate()
        }

    def _calculate_compliance_rate(self, window: int = 100) -> float:
        """
        Calculate recent compliance rate.

        Args:
            window: Number of recent evaluations to consider

        Returns:
            Compliance rate from 0.0 to 1.0
        """
        # This is simplified - in production, track all evaluations
        recent_violations = len(self.violations[-window:])
        if window == 0:
            return 1.0

        # Assume each violation represents 1 evaluation
        # In production, track total evaluations separately
        return max(0.0, 1.0 - (recent_violations / window))

    def update_policy_metadata(self, policy_name: str,
                              metadata: Dict[str, Any]) -> bool:
        """
        Update policy metadata/configuration.

        Args:
            policy_name: Policy to update
            metadata: New metadata values

        Returns:
            True if updated successfully
        """
        if policy_name in self.policies:
            self.policies[policy_name].metadata.update(metadata)
            logger.info("Policy metadata updated",
                       policy_name=policy_name,
                       metadata=metadata)
            return True
        return False

    def get_response_actions(self, recommendation: str, confidence: float) -> List[str]:
        """
        Get specific response actions based on recommendation.
        
        Args:
            recommendation: The security recommendation (allow, block, flag_for_review)
            confidence: The confidence score
            
        Returns:
            List of action strings
        """
        actions = []
        if recommendation == "block":
            actions.append("block_request")
        elif recommendation == "flag_for_review":
            actions.append("flag_for_review")
            
        if confidence > 0.95:
             actions.append("log_high_severity")
             
        return actions
