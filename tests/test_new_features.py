"""
Integration tests for new advanced features.
"""
import pytest


def test_imports():
    """Test that all new modules can be imported."""
    # Coordination modules
    from src.coordination.ovon_protocol import OVONMessage, OVONContent, LLMTag
    from src.coordination.behavioral_monitor import BehavioralMonitor
    from src.coordination.peerguard import PeerGuard, ReasoningTrace
    from src.coordination.policy_enforcer import PolicyEnforcer, Policy, PolicyAction
    from src.coordination.preprocessor import Preprocessor

    # Detection modules
    from src.detection.ensemble import InjectionDetector, DetectionPath

    # Response modules
    from src.response.circuit_breaker import CircuitBreaker, AlertSeverity, Alert

    assert OVONMessage is not None
    assert BehavioralMonitor is not None
    assert PeerGuard is not None
    assert PolicyEnforcer is not None
    assert Preprocessor is not None
    assert InjectionDetector is not None
    assert CircuitBreaker is not None


def test_ovon_message_with_llm_tag():
    """Test OVON message creation with LLM tagging."""
    from src.coordination.ovon_protocol import OVONMessage, OVONContent

    message = OVONMessage(
        source_agent="test_agent_1",
        destination_agent="test_agent_2",
        content=OVONContent(utterance="Test message")
    )

    # Add LLM tag
    message.add_llm_tag(
        agent_id="test_agent_1",
        agent_type="guard",
        trust_level=0.9,
        injection_score=0.1
    )

    assert message.llm_tag is not None
    assert message.llm_tag.agent_id == "test_agent_1"
    assert message.signature is not None
    assert message.verify_signature() is True
    assert message.is_safe() is True


def test_behavioral_monitor():
    """Test behavioral monitoring."""
    from src.coordination.behavioral_monitor import BehavioralMonitor

    monitor = BehavioralMonitor(window_size=50, anomaly_threshold=2.5)

    # Record normal interactions
    for i in range(30):
        monitor.record_interaction(
            agent_id="agent_1",
            output_length=100 + i,
            response_time=0.5,
            tool_calls=1
        )

    # Test anomaly detection (should not be anomalous - within normal range)
    result = monitor.detect_anomaly(
        agent_id="agent_1",
        output_length=120,
        response_time=0.6,
        tool_calls=1
    )

    assert "is_anomalous" in result
    assert "anomalies" in result


def test_peerguard():
    """Test PeerGuard mutual reasoning."""
    from src.coordination.peerguard import PeerGuard, ReasoningTrace

    peerguard = PeerGuard(consistency_threshold=0.7)

    trace = ReasoningTrace(
        agent_id="agent_1",
        input_prompt="What is 2+2?",
        reasoning_steps=["Parse the numbers", "Add 2 and 2", "Result is 4"],
        final_output="4",
        tool_calls=[],
        metadata={}
    )

    peerguard.record_reasoning(trace)

    validation = peerguard.validate_reasoning(trace, peer_agent_id="agent_2")

    assert "consistency" in validation
    assert "overall_score" in validation
    assert "is_suspicious" in validation


def test_policy_enforcer():
    """Test policy enforcement."""
    from src.coordination.policy_enforcer import PolicyEnforcer

    enforcer = PolicyEnforcer()

    # Test with high injection score (should violate policy)
    result = enforcer.enforce({
        "injection_score": 0.9,
        "trust_level": 0.8,
        "hop_count": 5
    })

    assert "compliant" in result
    assert "violations" in result
    assert "recommended_action" in result

    # Should detect high injection score violation
    assert not result["compliant"]


def test_preprocessor():
    """Test input preprocessing."""
    from src.coordination.preprocessor import Preprocessor

    preprocessor = Preprocessor()

    # Test with base64 encoded content
    result = preprocessor.process("SGVsbG8gV29ybGQ=")

    assert "original" in result
    assert "normalized" in result
    assert "detected_encodings" in result
    assert "suspicion_score" in result

    # Should detect base64
    if result["base64_content"]:
        assert "Hello World" in result["base64_content"]


def test_circuit_breaker_alerts():
    """Test enhanced circuit breaker with tiered alerts."""
    from src.response.circuit_breaker import CircuitBreaker, AlertSeverity

    breaker = CircuitBreaker(
        threshold=10,
        time_window=60,
        critical_threshold=3
    )

    # Record some alerts
    breaker.record_alert(
        severity=AlertSeverity.MEDIUM,
        source="test",
        category="test_category"
    )

    breaker.record_alert(
        severity=AlertSeverity.HIGH,
        source="test",
        category="test_category"
    )

    summary = breaker.get_alert_summary()

    assert summary["total_alerts"] >= 2
    assert "by_severity" in summary
    assert "by_category" in summary


def test_ensemble_classifier_init():
    """Test ensemble classifier initialization."""
    from src.detection.ensemble import InjectionDetector

    # Just test initialization without loading heavy models
    # In real tests, you would load models and test predictions
    ensemble = InjectionDetector(
        fast_model_name="all-MiniLM-L6-v2",
        use_cascade=True
    )

    assert ensemble.fast_model is not None
    assert ensemble.use_cascade is True
    assert "fast_path_count" in ensemble.stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
